"""
System Health Check Script
Verifies that all components are ready for deployment
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print(f"  ✓ {package_name} installed")
        return True
    except ImportError:
        print(f"  ✗ {package_name} NOT installed")
        return False

def main():
    print("="*70)
    print("DELHI FLOOD RISK SYSTEM - HEALTH CHECK")
    print("="*70)
    
    all_ok = True
    
    # Check data files
    print("\n[1/4] Checking data files...")
    all_ok &= check_file_exists('FinalTrainingData3.csv', 'Training data')
    all_ok &= check_file_exists('delhi_wards (1).json', 'GeoJSON boundaries')
    
    # Check code files
    print("\n[2/4] Checking code files...")
    all_ok &= check_file_exists('train_model.py', 'ML training script')
    all_ok &= check_file_exists('predict.py', 'Inference script')
    all_ok &= check_file_exists('index.html', 'Frontend map')
    all_ok &= check_file_exists('requirements.txt', 'Dependencies list')
    
    # Check Python packages
    print("\n[3/4] Checking Python packages...")
    packages = ['numpy', 'pandas', 'sklearn', 'joblib', 'matplotlib', 'seaborn']
    for pkg in packages:
        pkg_name = 'scikit-learn' if pkg == 'sklearn' else pkg
        installed = check_python_package(pkg)
        if not installed:
            all_ok = False
            print(f"      → Install with: pip install {pkg_name}")
    
    # Check generated files (optional)
    print("\n[4/4] Checking generated files (will be created after training)...")
    check_file_exists('flood_risk_model.pkl', 'Trained model')
    check_file_exists('ward_flood_predictions_2025.csv', '2025 predictions')
    check_file_exists('model_metrics.json', 'Performance metrics')
    
    # Final verdict
    print("\n" + "="*70)
    if all_ok:
        print("STATUS: ✓ SYSTEM READY")
        print("\nNext steps:")
        print("  1. Run: python train_model.py")
        print("  2. Open: index.html in your browser")
    else:
        print("STATUS: ✗ ISSUES DETECTED")
        print("\nPlease fix the issues above before proceeding.")
        print("Hint: Run 'pip install -r requirements.txt' to install packages")
    print("="*70)
    
    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
