#!/usr/bin/env python3
"""
Compare DataLoader startup time: 4 workers vs 6 workers
"""

import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

def test_workers(num_workers):
    """Test DataLoader with specific number of workers"""

    data_path = Path("results/optA_20251016_200330/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/train")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )

    # Measure first batch (worker spawning)
    start = time.time()
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    first_batch_time = time.time() - start

    # Measure second batch (workers running)
    start = time.time()
    batch = next(dataloader_iter)
    second_batch_time = time.time() - start

    # Measure third batch (prefetch active)
    start = time.time()
    batch = next(dataloader_iter)
    third_batch_time = time.time() - start

    return first_batch_time, second_batch_time, third_batch_time

def main():
    print("=" * 70)
    print("DATALOADER WORKERS COMPARISON: 4 vs 6")
    print("=" * 70)

    # Test with 6 workers (old setting)
    print("\n[TEST 1] Testing with 6 workers (OLD)...")
    time_6w_1, time_6w_2, time_6w_3 = test_workers(6)
    print(f"   First batch:  {time_6w_1:6.2f}s (worker spawn)")
    print(f"   Second batch: {time_6w_2:6.2f}s")
    print(f"   Third batch:  {time_6w_3:6.2f}s")

    # Test with 4 workers (new setting)
    print("\n[TEST 2] Testing with 4 workers (NEW)...")
    time_4w_1, time_4w_2, time_4w_3 = test_workers(4)
    print(f"   First batch:  {time_4w_1:6.2f}s (worker spawn)")
    print(f"   Second batch: {time_4w_2:6.2f}s")
    print(f"   Third batch:  {time_4w_3:6.2f}s")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    startup_improvement = ((time_6w_1 - time_4w_1) / time_6w_1) * 100

    print(f"\nStartup Time (First Batch):")
    print(f"   6 workers: {time_6w_1:6.2f}s")
    print(f"   4 workers: {time_4w_1:6.2f}s")
    print(f"   Improvement: {startup_improvement:5.1f}% faster ✅")

    print(f"\nTraining Speed (Second Batch):")
    print(f"   6 workers: {time_6w_2:6.2f}s")
    print(f"   4 workers: {time_4w_2:6.2f}s")
    if time_4w_2 < time_6w_2 * 1.1:  # Within 10% is acceptable
        print(f"   Status: Similar speed ✅ (no significant loss)")
    else:
        print(f"   Status: ⚠️ Slower with 4 workers")

    print(f"\nPrefetch Performance (Third Batch):")
    print(f"   6 workers: {time_6w_3:6.2f}s")
    print(f"   4 workers: {time_4w_3:6.2f}s")
    if time_4w_3 < time_6w_3 * 1.1:
        print(f"   Status: Similar speed ✅ (prefetch still works)")
    else:
        print(f"   Status: ⚠️ Slower with 4 workers")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if startup_improvement > 20 and time_4w_2 < time_6w_2 * 1.1:
        print("✅ OPTIMIZATION SUCCESSFUL!")
        print(f"   - Startup: {startup_improvement:.1f}% faster")
        print(f"   - Training: No significant speed loss")
        print(f"   - Recommendation: Use 4 workers")
    elif startup_improvement > 10:
        print("✅ MODERATE IMPROVEMENT")
        print(f"   - Startup: {startup_improvement:.1f}% faster")
        print(f"   - Recommendation: 4 workers acceptable")
    else:
        print("⚠️  MINIMAL IMPROVEMENT")
        print(f"   - Startup: Only {startup_improvement:.1f}% faster")
        print(f"   - Recommendation: Consider keeping 6 workers")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
