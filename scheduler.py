import subprocess
import time
from apscheduler.schedulers.background import BackgroundScheduler

def run_proforma_script():
    """Runs the proforma_s3store.py script."""
    try:
        subprocess.run(["path/to/venv/bin/python", "proforma_s3store.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running proforma_s3store.py: {e}")

def run_po_script():
    """Runs the PO_s3store.py script."""
    try:
        subprocess.run(["path/to/venv/bin/python", "PO_s3store.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running PO_s3store.py: {e}")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_proforma_script, 'cron', hour=0, minute=0)
scheduler.add_job(run_po_script, 'cron', hour=0, minute=0)
scheduler.start()

print("Scheduler started. Running tasks daily at 12:00 AM.")

# Keep the script running
try:
    while True:
        time.sleep(60)  # Keep the script alive
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    print("Scheduler stopped.")


#This script runs proforma_s3store.py and PO_s3store.py at 12:00 AM daily.
#It keeps running in the background to trigger the jobs at the scheduled time.
#If new files arrive in the email, the existing scripts will process them automatically.