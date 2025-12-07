#!/usr/bin/env python3
"""
Profile DataLoader startup overhead to find exact bottleneck
"""

import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

def profile_dataloader():
    """Profile each stage of DataLoader initialization"""

    # Use actual classification crop dataset path
    data_path = Path("results/optA_20251016_200330/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/train")

    print("=" * 60)
    print("DATALOADER PROFILING")
    print("=" * 60)

    # Stage 1: Dataset initialization
    print("\n[STAGE 1] Dataset Initialization...")
    start = time.time()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    init_time = time.time() - start
    print(f"   Time: {init_time:.2f}s")
    print(f"   Images: {len(dataset)}")
    print(f"   Classes: {dataset.classes}")

    # Stage 2: DataLoader creation (with workers)
    print("\n[STAGE 2] DataLoader Creation (6 workers)...")
    start = time.time()

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    create_time = time.time() - start
    print(f"   Time: {create_time:.2f}s")

    # Stage 3: First batch (worker spawning + first load)
    print("\n[STAGE 3] First Batch (Worker Spawn + Load)...")
    start = time.time()

    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    first_batch_time = time.time() - start
    print(f"   Time: {first_batch_time:.2f}s")
    print(f"   Batch shape: {batch[0].shape}")

    # Stage 4: Second batch (workers already running)
    print("\n[STAGE 4] Second Batch (No Overhead)...")
    start = time.time()

    batch = next(dataloader_iter)

    second_batch_time = time.time() - start
    print(f"   Time: {second_batch_time:.2f}s")

    # Stage 5: Third batch (prefetch active)
    print("\n[STAGE 5] Third Batch (Prefetch Active)...")
    start = time.time()

    batch = next(dataloader_iter)

    third_batch_time = time.time() - start
    print(f"   Time: {third_batch_time:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset init:        {init_time:6.2f}s")
    print(f"DataLoader create:   {create_time:6.2f}s")
    print(f"First batch (spawn): {first_batch_time:6.2f}s  ← EXPECTED SLOWEST")
    print(f"Second batch:        {second_batch_time:6.2f}s")
    print(f"Third batch:         {third_batch_time:6.2f}s")
    print(f"\nTotal startup:       {init_time + create_time + first_batch_time:6.2f}s")
    print(f"Speedup 2nd→3rd:     {second_batch_time/third_batch_time:6.2f}x")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if init_time > 5:
        print(f"⚠️  Dataset init SLOW ({init_time:.1f}s) → Check disk speed (HDD vs SSD)")
    else:
        print(f"✅ Dataset init OK ({init_time:.1f}s)")

    if first_batch_time > 15:
        print(f"⚠️  First batch VERY SLOW ({first_batch_time:.1f}s) → Worker spawning overhead!")
        print("   Possible causes:")
        print("   - Windows multiprocessing (spawn vs fork)")
        print("   - Slow disk I/O (HDD)")
        print("   - Antivirus scanning")
        print("   - Too many workers (6 might be too much)")
    elif first_batch_time > 5:
        print(f"⚠️  First batch slow ({first_batch_time:.1f}s) → Normal for Windows")
    else:
        print(f"✅ First batch fast ({first_batch_time:.1f}s)")

    if second_batch_time > 1:
        print(f"⚠️  Second batch slow ({second_batch_time:.1f}s) → Disk I/O bottleneck!")
    else:
        print(f"✅ Second batch fast ({second_batch_time:.1f}s)")

    if third_batch_time < second_batch_time * 0.8:
        print(f"✅ Prefetching working ({third_batch_time:.1f}s vs {second_batch_time:.1f}s)")
    else:
        print(f"⚠️  Prefetching NOT helping → Disk bottleneck or transform overhead")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    profile_dataloader()
