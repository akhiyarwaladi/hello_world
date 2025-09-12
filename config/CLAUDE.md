# Configuration Directory - Context for Claude

## Purpose
Contains all configuration files for the malaria detection pipeline.

## Files
- `dataset_config.yaml` - Dataset sources, augmentation settings, and class mappings
- `class_names.yaml` - Unified class names for 6-class malaria classification
- Additional YAML configs as needed

## Key Configurations
- **6 Classes**: P_falciparum, P_vivax, P_malariae, P_ovale, Mixed_infection, Uninfected
- **Datasets**: NIH Cell, MP-IDB, BBBC041, PlasmoID, IML, Uganda datasets
- **Augmentation**: Rotation, flip, brightness, contrast adjustments for minority classes

## Current Status (Updated: December 12, 2024)
- ✅ **Configurations active** - Scripts using YAML configs for pipeline execution
- 🔄 **Species mapping FIXED** - Updated class mappings working in preprocessing
- 📊 **Training configs** - Supporting multiple YOLOv8 training sessions

## Usage
Scripts automatically load configs from this directory using YAML parsing.