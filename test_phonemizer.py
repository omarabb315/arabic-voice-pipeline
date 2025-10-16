"""
Test script to verify phonemizer works correctly with Arabic text.
Run this before starting the full pipeline to ensure phonemization is working.
"""

from phonemizer.backend import EspeakBackend


def test_arabic_phonemizer():
    """Test phonemizer with various Arabic texts."""
    
    print("=" * 70)
    print("Testing Arabic Phonemizer")
    print("=" * 70)
    print()
    
    # Initialize phonemizer
    print("Initializing EspeakBackend...")
    try:
        # Try to find espeak-ng executable
        import subprocess
        import os
        
        # Check if espeak-ng is available
        try:
            result = subprocess.run(['which', 'espeak-ng'], capture_output=True, text=True, check=True)
            espeak_path = result.stdout.strip()
            print(f"Found espeak-ng at: {espeak_path}")
            
            # Set environment variable for phonemizer
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeak_path
        except:
            pass
        
        g2p = EspeakBackend(
            language='ar',  # Arabic
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            language_switch="remove-flags"
        )
        print("✅ Phonemizer initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize phonemizer: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure espeak-ng is installed: brew install espeak-ng")
        print("  2. Try creating a symlink: ln -s /opt/homebrew/bin/espeak-ng /opt/homebrew/bin/espeak")
        print("  3. Or set PHONEMIZER_ESPEAK_PATH=/opt/homebrew/bin/espeak-ng")
        return False
    
    print()
    
    # Test cases - various Arabic texts
    test_texts = [
        "مرحباً بكم",
        "كيف حالكْ اليوم؟",
        "الطقس جميل",
        "وِين وِين رَايِح؟",  # Gulf dialect (from your example)
        "أهلاً وسهلاً",
        "شكراً جزيلاً",
        "الحمد لله",
        "السَّلام عليكمُ ورحمة الله وبركاته",
    ]
    
    print("Testing Arabic texts:")
    print("-" * 70)
    
    all_passed = True
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        
        try:
            # Phonemize
            phones = g2p.phonemize([text])
            
            # Check result
            if not phones or not phones[0]:
                print(f"   ⚠️  WARNING: Empty phonemization output!")
                all_passed = False
            else:
                phones_str = ' '.join(phones[0].split())
                print(f"   Phones: {phones_str}")
                print(f"   Length: {len(phones[0].split())} phonemes")
                print(f"   ✅ Success")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ All tests passed!")
        print()
        print("Your phonemizer is working correctly with Arabic.")
        print("You can proceed with the data preparation pipeline.")
    else:
        print("⚠️  Some tests failed!")
        print()
        print("This might cause issues during training.")
        print("Please check the errors above and fix them before proceeding.")
    
    print("=" * 70)
    
    return all_passed


def test_multiple_languages():
    """Test if you need multi-language support."""
    
    print()
    print("=" * 70)
    print("Testing Multi-Language Support (Optional)")
    print("=" * 70)
    print()
    
    # Test if your data might have mixed languages
    mixed_texts = [
        "Hello مرحباً",  # English + Arabic
        "مرحباً in Dubai",  # Arabic + English
    ]
    
    print("If your data contains mixed language text:")
    print("-" * 70)
    
    for text in mixed_texts:
        print(f"\nText: {text}")
        
        try:
            # Test with Arabic backend
            g2p_ar = EspeakBackend(
                language='ar',
                preserve_punctuation=True,
                with_stress=True,
            )
            phones = g2p_ar.phonemize([text])
            phones_str = ' '.join(phones[0].split()) if phones and phones[0] else "EMPTY"
            print(f"   Arabic backend: {phones_str}")
        
        except Exception as e:
            print(f"   Error: {e}")
    
    print()
    print("Note: If you have mixed language data, you may need to:")
    print("  1. Filter out mixed language samples")
    print("  2. Or use language detection and separate processing")


def check_espeak_installation():
    """Check if espeak is properly installed."""
    
    print()
    print("=" * 70)
    print("Checking espeak Installation")
    print("=" * 70)
    print()
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['espeak', '--version'],
            capture_output=True,
            text=True
        )
        print("✅ espeak is installed:")
        print(result.stdout)
        
        # Check available languages
        result = subprocess.run(
            ['espeak', '--voices'],
            capture_output=True,
            text=True
        )
        
        if 'ar' in result.stdout or 'arabic' in result.stdout.lower():
            print("✅ Arabic language support is available")
        else:
            print("⚠️  Arabic language might not be available")
            print("   Install espeak-ng for better Arabic support:")
            print("   - macOS: brew install espeak-ng")
            print("   - Ubuntu: sudo apt-get install espeak-ng")
        
    except FileNotFoundError:
        print("❌ espeak is not installed!")
        print()
        print("Installation instructions:")
        print("  - macOS: brew install espeak-ng")
        print("  - Ubuntu: sudo apt-get install espeak-ng")
        print("  - CentOS: sudo yum install espeak-ng")
        print()
        return False
    
    return True


if __name__ == "__main__":
    print()
    print("🔧 Arabic Phonemizer Test Utility")
    print()
    
    # Step 1: Check espeak installation
    if not check_espeak_installation():
        print()
        print("Please install espeak first, then run this test again.")
        exit(1)
    
    # Step 2: Test Arabic phonemization
    success = test_arabic_phonemizer()
    
    # Step 3: Test multi-language (optional)
    test_multiple_languages()
    
    print()
    
    if success:
        print("🎉 Ready to proceed with the pipeline!")
        exit(0)
    else:
        print("⚠️  Please fix the issues above before proceeding.")
        exit(1)

