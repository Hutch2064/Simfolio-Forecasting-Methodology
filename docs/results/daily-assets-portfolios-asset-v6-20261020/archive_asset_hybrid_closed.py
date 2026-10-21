"""Coordinate closed full-result preservation with the shared forecast slots."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys
O=Path(__file__).resolve().parent
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
with (O/'asset_v6_forecast_slots.lock').open('a') as lease:
    fcntl.flock(lease,fcntl.LOCK_EX)
    write(O/'asset_hybrid_archive_active.json',{'pid':os.getpid(),'status':'running','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'forecast_pool_lease_held_for_bulk_IO':True})
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.8','--artifact-root',str(O),'--budget-gib','12'],check=True)
    subprocess.run([sys.executable,str(O/'preserve_asset_hybrid_closed.py')],cwd=O,check=True)
    write(O/'asset_hybrid_archive_active.json',{'pid':os.getpid(),'status':'terminal','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_originals_removed':False})
