"""
Fine-tune Faster R-CNN on V-COCO

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import math
import json
import copy
import time
import torch
import bisect
import argparse
import torchvision
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from PIL import Image
from itertools import repeat, chain
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.sampler import Sampler, BatchSampler

import pocket
from pocket.ops import RandomHorizontalFlip, to_tensor

sys.path.append('..')
from vcoco import VCOCO

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => 
        tensor([0, 1, 3])
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

class DetectorEngine(pocket.core.DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self._val_loader = val_loader

    def _on_start_epoch(self):
        self._state.epoch += 1
        self._state.net.train()
        self._train_loader.batch_sampler.sampler.set_epoch(self._state.epoch)

    def _on_each_iteration(self):
        self._state.output = self._state.net(*self._state.inputs, targets=self._state.targets)
        self._state.loss = sum(loss for loss in self._state.output.values())
        self._state.optimizer.zero_grad()
        self._state.loss.backward()
        self._state.optimizer.step()

    def _on_end_epoch(self):
        timer = pocket.utils.HandyTimer(maxlen=1)
        with timer:
            ap, max_rec = self.validate()
        if self._rank == 0:
            print("\n=> Validation (+{:.2f})\n"
                "Epoch: {} | mAP: {:.4f}, mRec: {:.4f} | Time: {:.2f}s\n".format(
                    time.time() - self._dawn, self._state.epoch,
                    ap.mean().item(), max_rec.mean().item(), timer[0]
                ))
        super()._on_end_epoch()

    @torch.no_grad()
    def validate(self, min_iou=0.5, nms_thresh=0.5):
        num_gt = torch.zeros(80)
        associate = pocket.utils.BoxAssociation(min_iou=min_iou)
        meter = pocket.utils.DetectionAPMeter(
            80, algorithm='INT', nproc=10
        )
        self._state.net.eval()
        for batch in tqdm(self._val_loader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = self._state.net(inputs)
            assert len(output) == 1, "The batch size should be one"
            # Relocate back to cpu
            output = pocket.ops.relocate_to_cpu(output[0])
            target = batch[1][0]

            gt_boxes = target['boxes']
            gt_classes = target['labels']

            for c in gt_classes:
                num_gt[c - 1] += 1

            # Associate detections with ground truth
            binary_labels = torch.zeros_like(output['scores'])
            unique_obj = output['labels'].unique()
            for obj_idx in unique_obj:
                det_idx = torch.nonzero(output['labels'] == obj_idx).squeeze(1)
                gt_idx = torch.nonzero(gt_classes == obj_idx).squeeze(1)
                if len(gt_idx) == 0:
                    continue
                binary_labels[det_idx] = associate(
                    gt_boxes[gt_idx].view(-1, 4),
                    output['boxes'][det_idx].view(-1, 4),
                    output['scores'][det_idx].view(-1)
                )

            # Synchronise the results
            all_results = np.stack([
                output['scores'].numpy(),
                output['labels'].numpy(),
                binary_labels.numpy()
            ])
            all_results_sync = pocket.utils.all_gather(all_results)
            if self._rank == 0:
                scores, pred, labels = torch.from_numpy(
                    np.concatenate(all_results_sync, axis=1)
                ).unbind(0)
                meter.append(scores, pred - 1, labels)

        # Sync the number of ground truth instances
        num_gt = num_gt.cuda()
        dist.barrier()
        dist.all_reduce(num_gt)
        # Compute mAP in the master process
        if self._rank == 0:
            meter.num_gt = num_gt.tolist()
            ap = meter.eval()
            return ap, meter.max_rec
        else:
            return None, None

class VCOCOObject(Dataset):
    def __init__(self, dataset, random_flip=False):
        self.dataset = dataset
        self.transform = RandomHorizontalFlip() if random_flip else None
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        labels = torch.cat([
            torch.ones_like(target['objects']),
            target['objects']
        ])

        keep_idx = unique(boxes, dim=0)
        boxes = boxes[keep_idx].view(-1, 4)
        labels = labels[keep_idx].view(-1)

        image = to_tensor(image, input_format='pil')
        return [image], [dict(boxes=boxes, labels=labels)]

def collate_fn(batch):
    images = []
    targets = []
    for im, tar in batch:
        images += im
        targets += tar
    return images, targets

"""
Batch sampler that groups images by aspect ratio
https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
"""

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size    

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(aspect_ratios, k=0):
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)
    torch.manual_seed(args.random_seed)

    trainval = VCOCOObject(VCOCO(
        root=os.path.join(args.data_root, "mscoco2014/train2014"),
        anno_file=os.path.join(args.data_root, "instances_vcoco_trainval.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ), random_flip=True)
    test = VCOCOObject(VCOCO(
        root=os.path.join(args.data_root, "mscoco2014/val2014"),
        anno_file=os.path.join(args.data_root, "instances_vcoco_test.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ))

    # Prepare distributed sampler
    train_sampler = DistributedSampler(
        trainval, num_replicas=args.world_size, rank=rank
    )
    val_sampler = DistributedSampler(
        test, num_replicas=args.world_size, rank=rank
    )
    # Prepare grouped batch sampler
    def div(a, b):
        return a / b
    aspect_ratios = [div(*trainval.dataset.image_size(i)) for i in range(len(trainval))]
    group_ids = create_aspect_ratio_groups(aspect_ratios, k=args.aspect_ratio_group_factor)

    train_loader = DataLoader(
        dataset=trainval, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
        batch_sampler=GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size)
    )

    val_loader = DataLoader(
        dataset=test, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
        sampler=val_sampler, batch_size=1
    )

    net = pocket.models.fasterrcnn_resnet_fpn('resnet50', pretrained=True)
    net.cuda()
    
    engine = DetectorEngine(
        net, train_loader, val_loader,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir,
        optim_params=dict(
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        ),
        lr_scheduler=True,
        lr_sched_params=dict(
            milestones=args.milestones,
            gamma=args.lr_decay
        )
    )

    engine(args.num_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on V-COCO")
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--num-epochs', default=15, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--milestones', nargs='+', default=[8, 12], type=int)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))