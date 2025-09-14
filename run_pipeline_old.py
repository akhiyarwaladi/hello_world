#!/usr/bin/env python3
"""
Simple pipeline runner - wrapper untuk pipeline_manager.py
Menggantikan quick_setup_new_machine.sh dengan cara yang lebih sederhana
"""

from pipeline_manager import MalariaPipelineManager

def main():
    """Run pipeline with simple interface"""

    pipeline = MalariaPipelineManager()

    print("🔬 MALARIA DETECTION PIPELINE")
    print("="*60)
    print("Ini akan menjalankan pipeline lengkap dengan checkpoint support.")
    print("Jika tahap sudah selesai, akan di-skip otomatis.\n")

    # Show current status first
    pipeline.show_pipeline_status()

    print("\nOptions:")
    print("1. Lanjutkan pipeline (skip completed stages)")
    print("2. Restart dari awal (hapus semua checkpoint)")
    print("3. Lihat status saja")
    print("4. Keluar")

    choice = input("\nPilih (1-4) [1]: ").strip() or "1"

    if choice == "1":
        print("\n🚀 Melanjutkan pipeline...")
        success = pipeline.run_pipeline()

    elif choice == "2":
        print("\n⚠️  Restart dari awal - semua checkpoint akan dihapus!")
        confirm = input("Yakin? (y/N): ").strip().lower()
        if confirm == 'y':
            success = pipeline.run_pipeline(force_restart=True)
        else:
            print("Dibatalkan.")
            return

    elif choice == "3":
        pipeline.show_pipeline_status()
        return

    elif choice == "4":
        print("Keluar.")
        return

    else:
        print("Pilihan tidak valid.")
        return

    if success:
        print("\n🎉 Pipeline selesai! Siap untuk training model.")
    else:
        print("\n❌ Pipeline berhenti karena error. Periksa log untuk detail.")

if __name__ == "__main__":
    main()