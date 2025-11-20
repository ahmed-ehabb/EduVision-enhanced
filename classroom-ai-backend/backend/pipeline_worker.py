"""
Pipeline Worker Script
======================

Standalone script to run the Teacher Module V2 pipeline in a subprocess.
This avoids issues with multiprocessing + CUDA on Windows.

Author: Ahmed
Date: 2025-11-06
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path

# Add backend to path - handle both direct execution and subprocess execution
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, script_dir)

from backend.teacher_module_v2 import TeacherModuleV2


def setup_logging(log_file: str):
    """Set up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Run the pipeline with arguments from command line."""
    # Print to stdout immediately to verify subprocess starts
    print(f"[WORKER] Starting worker process PID={os.getpid()}", flush=True)
    print(f"[WORKER] Python: {sys.version}", flush=True)
    print(f"[WORKER] CWD: {os.getcwd()}", flush=True)
    print(f"[WORKER] Args: {sys.argv}", flush=True)

    if len(sys.argv) != 2:
        print("Usage: python pipeline_worker.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    print(f"[WORKER] Loading config from: {config_file}", flush=True)

    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"[WORKER] Config loaded successfully", flush=True)
    except Exception as e:
        print(f"[WORKER] ERROR loading config: {e}", flush=True)
        sys.exit(1)

    lecture_id = config['lecture_id']
    audio_path = config['audio_path']
    textbook_paragraphs = config['textbook_paragraphs']
    lecture_title = config['lecture_title']
    pdf_path = config.get('pdf_path')
    results_file = config['results_file']
    log_file = results_file.replace('.json', '_log.txt')

    logger = setup_logging(log_file)

    try:
        logger.info(f"[WORKER] Started for lecture {lecture_id}")
        logger.info(f"[WORKER] PID: {os.getpid()}")
        logger.info(f"[WORKER] Audio path: {audio_path}")
        logger.info(f"[WORKER] Results file: {results_file}")

        logger.info("[WORKER] Importing TeacherModuleV2...")
        # Import is already done above
        logger.info("[WORKER] Import successful")

        logger.info("[WORKER] Creating TeacherModuleV2 instance...")
        teacher = TeacherModuleV2()
        logger.info("[WORKER] TeacherModuleV2 instance created")

        logger.info("[WORKER] Starting pipeline processing...")
        results = teacher.process_lecture(
            audio_path=audio_path,
            textbook_paragraphs=textbook_paragraphs,
            lecture_title=lecture_title,
            pdf_path=pdf_path
        )
        logger.info("[WORKER] Pipeline processing complete")

        logger.info(f"[WORKER] Writing results to {results_file}")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("[WORKER] Results written successfully")

        # Clean up config file
        if os.path.exists(config_file):
            os.unlink(config_file)

        logger.info("[WORKER] Worker completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"[WORKER] ERROR: {str(e)}")
        logger.error(f"[WORKER] Traceback:\n{traceback.format_exc()}")

        # Save error to file
        error_result = {
            "success": False,
            "errors": [str(e)],
            "traceback": traceback.format_exc(),
            "fatal_error": True
        }

        try:
            with open(results_file, 'w') as f:
                json.dump(error_result, f, indent=2)
            logger.info(f"[WORKER] Error result written to {results_file}")
        except Exception as write_error:
            logger.error(f"[WORKER] Failed to write error result: {write_error}")

        # Clean up config file
        if os.path.exists(config_file):
            os.unlink(config_file)

        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[WORKER] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
