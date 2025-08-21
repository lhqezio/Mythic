# Mythic Project Structure

This document describes the restructured Mythic AI Engine project layout.

## 📁 Directory Structure

```
Mythic/
├── .gitignore              # Git ignore patterns for the restructured project
├── LICENSE                 # Project license
├── README.md               # Main project overview and quick start
├── PROJECT_STRUCTURE.md    # This file - project structure documentation
├── demo/                   # Complete demo application
│   ├── README.md          # Comprehensive demo documentation
│   ├── llm_to_tts_demo_enhanced.py  # Main demo script
│   ├── config.py          # Configuration management
│   ├── run_demo.sh        # Demo launcher script
│   ├── demo_setup.sh      # Environment setup script
│   ├── requirements.txt    # Python dependencies
│   ├── fix_cuda_llama_cpp.sh  # CUDA setup helper
│   ├── output/            # Generated audio files (gitignored)
│   ├── runtimes/          # Runtime models and files (gitignored)
│   ├── .venv/             # Virtual environment (gitignored)
│   └── demo.log           # Demo execution logs (gitignored)
└── docs/                   # Technical documentation
    └── tech-design.md      # Architecture and implementation details
```

## 🔄 What Changed

### Before (Old Structure)
- Demo files were scattered throughout the project
- No clear separation between core engine and demo code
- Confusing path references

### After (New Structure)
- **`demo/`** - Self-contained demo application
- **`docs/`** - Technical documentation
- **Root level** - Clean project overview and navigation
- Clear separation of concerns
- Consistent path references

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mythic
   ```

2. **Run the demo**
   ```bash
   cd demo
   chmod +x run_demo.sh demo_setup.sh
   ./demo_setup.sh
   ./run_demo.sh
   ```

3. **Explore the code**
   - Start with `demo/README.md` for demo usage
   - Check `docs/tech-design.md` for technical details
   - Review `demo/config.py` for customization options

## 🎯 Key Benefits

- **Clear Organization**: Demo code is self-contained and easy to find
- **Easy Setup**: Single command to get the demo running
- **Better Documentation**: Separate docs for different user types
- **Cleaner Repository**: Core project files are separate from demo artifacts
- **Easier Development**: Clear structure for adding new features

## 🔧 Development Workflow

1. **Core Engine Development**: Work in the root directory
2. **Demo Development**: Work in the `demo/` directory
3. **Documentation**: Update files in `docs/` directory
4. **Testing**: Use the demo as a testbed for new features

## 📝 File Descriptions

### Root Level Files
- **`.gitignore`**: Updated for restructured project
- **`README.md`**: Project overview and quick start guide
- **`PROJECT_STRUCTURE.md`**: This documentation file

### Demo Directory
- **`run_demo.sh`**: Enhanced demo launcher with performance profiles
- **`demo_setup.sh`**: Automated environment setup
- **`config.py`**: Centralized configuration management
- **`llm_to_tts_demo_enhanced.py`**: Main demo implementation

### Documentation
- **`docs/tech-design.md`**: Technical architecture and implementation
- **`demo/README.md`**: Comprehensive demo usage guide

## 🆘 Need Help?

- Check the demo README: `demo/README.md`
- Review technical docs: `docs/tech-design.md`
- Look at the main README: `README.md`
- Examine the project structure: `PROJECT_STRUCTURE.md`

---

**The restructured project makes it easier than ever to get started with Mythic! 🎉**
