#!/usr/bin/env python3
"""
The Gardener Environment Test
Tests the basic functionality of The Gardener's game environment

Origin: Protocol 37 - The Move 37 Protocol
Purpose: Verify environment setup and basic operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import SanctuaryEnvironment

def test_environment_initialization():
    """Test that the environment initializes correctly"""
    print("🌱 Testing Gardener Environment Initialization...")
    
    try:
        env = SanctuaryEnvironment()
        print("✅ Environment initialized successfully")
        print(f"   Repository path: {env.repo_path}")
        print(f"   Git available: {hasattr(env.git, 'use_gitpython')}")
        return True
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return False

def test_reset_functionality():
    """Test environment reset"""
    print("\n🔄 Testing Environment Reset...")
    
    try:
        env = SanctuaryEnvironment()
        observation = env.reset()
        print("✅ Environment reset successful")
        print(f"   Current branch: {observation.get('current_branch', 'unknown')}")
        print(f"   Protocols count: {observation.get('protocols_count', 0)}")
        return True
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False

def test_file_reading():
    """Test file reading action"""
    print("\n📖 Testing File Reading Action...")
    
    try:
        env = SanctuaryEnvironment()
        env.reset()
        
        # Test reading a protocol file (allowed path)
        obs, reward, done, info = env.step(0, file_path="01_PROTOCOLS/00_Prometheus_Protocol.md")
        
        if info.get('success'):
            print("✅ File reading successful")
            print(f"   Reward: {reward}")
            print(f"   Content preview: {info.get('content_preview', '')[:50]}...")
        else:
            print(f"❌ File reading failed: {info.get('error', 'Unknown error')}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ File reading test failed: {e}")
        return False

def test_git_operations():
    """Test git wrapper functionality"""
    print("\n🔧 Testing Git Operations...")
    
    try:
        env = SanctuaryEnvironment()
        
        # Test git status
        status = env.git.get_status()
        print("✅ Git status retrieved")
        print(f"   Modified files: {len(status['modified'])}")
        print(f"   Untracked files: {len(status['untracked'])}")
        
        # Test current branch
        branch = env.git.get_current_branch()
        print(f"   Current branch: {branch}")
        
        return True
    except Exception as e:
        print(f"❌ Git operations test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 The Gardener Environment Test Suite")
    print("=" * 50)
    
    tests = [
        test_environment_initialization,
        test_reset_functionality,
        test_file_reading,
        test_git_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Gardener is ready to tend the Cognitive Genome.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
