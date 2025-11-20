"""
Classroom AI Assistant - Main Entry Point

A comprehensive AI-powered system with two-phase processing:
Phase 1: Audio processing (ASR, translation, notes, alignment, engagement)
Phase 2: On-demand quiz generation with optimized GPU memory

Usage:
    python main.py

The application will start on http://localhost:8001 and automatically
open the dashboard in your default browser.
"""

import os
import sys
import webbrowser
import time
import threading
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure backend modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import GPU configuration first
from backend.gpu_config import setup_gpu_environment, print_device_status, optimize_model_loading
from backend.gpu_memory_manager import gpu_memory_manager
import backend.status_manager as sm

def initialize_phase_one_models():
    """Phase 1: Initialize main audio processing models"""
    print("🎵 PHASE 1: INITIALIZING MAIN PROCESSING MODELS")
    print("=" * 60)
    
    # Setup GPU environment
    setup_gpu_environment()
    optimize_model_loading()
    print_device_status()
    
    # Get memory status before loading
    if gpu_memory_manager.get_gpu_memory_info()['total'] > 0:
        memory_info = gpu_memory_manager.get_gpu_memory_info()
        print(f"📊 Available GPU memory: {memory_info['available']:.2f} GB")
    
    # The actual model loading will happen when the server starts.
    # We can't check the status until the server's lifespan manager runs.
    # This section will now just print what is *expected* to be loaded.
    print("\n🔧 Main processing models will be loaded on server startup.")
    
    # We can't access the status flags before the server starts,
    # so this check is removed. The server log will show the real status.
    
    # Show final memory usage after Phase 1 - This is also speculative
    if gpu_memory_manager.get_gpu_memory_info()['total'] > 0:
        memory_info = gpu_memory_manager.get_gpu_memory_info()
        print(f"\n📊 GPU memory before model loading: {memory_info['used']:.2f} GB used, {memory_info['available']:.2f} GB available")
    
    # Print Phase 1 status based on what should happen
    print("\n✅ PHASE 1 COMPLETE - Main Processing Models will be Ready:")
    print(f"  🎤 ASR (Speech Recognition): {'✅ Will be available'}")
    print(f"  📝 Notes Generation: {'✅ Will be available'}")
    print(f"  🌐 Translation: {'✅ Will be available'}")
    print(f"  📊 Text Alignment: {'✅ Will be available'}")
    print(f"  📈 Engagement Analysis: {'✅ Will be available'}")
    
    # We return a placeholder status because the real status is not known yet.
    return {
        'asr': True, 'notes': True, 'translation': True,
        'text_alignment': True, 'engagement': True, 'quiz_ready': True
    }

def show_phase_two_info():
    """Show information about Phase 2: Quiz Generation"""
    print("\n🎯 PHASE 2: ON-DEMAND QUIZ GENERATION")
    print("=" * 60)
    
    # This check is now optimistic. The server /status endpoint is the source of truth.
    print("✅ Quiz Generation: Available (loads on-demand)")
    print("🎯 Quiz model will use optimized GPU memory allocation:")
    print("  • Main models automatically moved to CPU when quiz is requested")
    print("  • Quiz model gets priority access to GPU memory")
    print("  • Supports 4-bit quantized model for optimal performance")
    print("  • Automatic fallback to CPU if GPU memory insufficient")
    
    # Show available memory for quiz
    if gpu_memory_manager.get_gpu_memory_info()['total'] > 0:
        # Simulate memory preparation
        temp_memory = gpu_memory_manager.get_gpu_memory_info()
        estimated_available = temp_memory['available'] + temp_memory['used'] * 0.8  # Estimate after freeing main models
        print(f"📊 Estimated memory for quiz model: ~{estimated_available:.2f} GB")
        
        if estimated_available >= 2.5:
            print("✅ Sufficient memory for quantized quiz model on GPU")
        else:
            print("⚠️ Limited memory - quiz model may use CPU")

    print("\n🔧 Quiz Generation Workflow:")
    print("  1. User uploads audio → Phase 1 models process it")
    print("  2. User requests quiz → GPU memory optimized for quiz model")
    print("  3. Quiz model loads with maximum available GPU memory")
    print("  4. High-quality quiz questions generated")

def display_system_summary(phase1_status):
    """Display final system summary"""
    print("\n" + "=" * 60)
    print("🎉 TWO-PHASE CLASSROOM AI SYSTEM READY")
    print("=" * 60)
    
    # Count available features
    available_features = sum(1 for status in phase1_status.values() if status)
    total_features = len(phase1_status)
    
    print(f"📊 System Status: {available_features}/{total_features} features available")
    
    if available_features == total_features:
        print("🎉 All features fully operational!")
    elif available_features >= 4:
        print("✅ System operational with most features available")
    else:
        print("⚠️ System operational with limited features")
    
    print("\n🚀 Ready for:")
    print("  🎵 Audio file processing (transcription, translation, notes)")
    print("  📊 Text alignment and engagement analysis")
    print("  🎯 On-demand quiz generation with GPU optimization")
    print("  🌐 Web dashboard at http://localhost:8001")
    
    print("\n💡 Usage Tips:")
    print("  • Upload audio files through the web dashboard")
    print("  • Processing results appear immediately (Phase 1)")
    print("  • Request quiz generation with custom parameters (Phase 2)")
    print("  • Monitor GPU memory usage through /gpu-memory-status endpoint")

def open_dashboard():
    """Open the dashboard in the default browser"""
    print("🌐 Opening dashboard...")
    time.sleep(2)
    try:
        # Assuming the frontend is served at the root or a specific page
        webbrowser.open("http://localhost:3000") # Corrected to standard React port
        print("🌐 Dashboard should be open at http://localhost:3000")
    except Exception as e:
        print(f"⚠️ Could not open browser: {e}")
        print("📋 Please manually open: http://localhost:3000")

def main():
    """Main application entry point with two-phase initialization"""
    print("🎓 CLASSROOM AI ASSISTANT - TWO-PHASE SYSTEM")
    print("=" * 60)
    print("Optimized GPU memory management for maximum performance")
    print("=" * 60)
    
    # Start the server in a separate thread so we can run checks after it starts
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for the server to begin starting up and loading models
    print("\n⏳ Waiting for server to initialize and load models...")
    time.sleep(15) # Increased wait time for model loading

    # Now that the server is starting, we can check the status from the status_manager
    print_system_status()

    # The rest of the logic continues in the server thread.
    # We can open the browser here.
    open_dashboard()
    
    # Keep the main thread alive, otherwise the daemon server thread will exit
    server_thread.join()

def start_server():
    """Imports and starts the FastAPI server."""
    try:
        from backend.api_server import app
        import uvicorn
        
        print(f"\n🌐 Starting server at: http://localhost:8001")
        print("\n📋 For API docs, visit: http://localhost:8001/docs")
        
        uvicorn.run(app, host="localhost", port=8001, log_level="info")

    except Exception as e:
        print(f"❌ Error during server execution: {e}")
        sys.exit(1)

def print_system_status():
    """Prints the final status of the system by checking the status manager."""
    print("\n" + "=" * 60)
    print("✅ SYSTEM INITIALIZATION COMPLETE")
    print("=" * 60)

    status = {
        "ASR": sm.ASR_AVAILABLE,
        "Notes": sm.NOTES_GENERATOR_AVAILABLE,
        "Translation": sm.TRANSLATOR_AVAILABLE,
        "Alignment": sm.TEXT_ALIGNMENT_AVAILABLE,
        "Quiz": sm.QUIZ_GENERATOR_AVAILABLE,
        "Database": sm.DATABASE_AVAILABLE
    }
    
    available_features = sum(1 for v in status.values() if v)
    total_features = len(status)
    
    print(f"📊 System Status: {available_features}/{total_features} features available")
    for feature, is_available in status.items():
        print(f"  - {feature}: {'✅ Available' if is_available else '❌ Unavailable'}")

    if available_features == total_features:
        print("\n🎉 All features fully operational!")
    else:
        print("\n⚠️ System operational with some features disabled. Check server logs for errors.")

    print("\n🚀 Ready for use.")
    print("  🌐 Web dashboard at http://localhost:8001")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ A critical error occurred in main: {e}")
        sys.exit(1) 