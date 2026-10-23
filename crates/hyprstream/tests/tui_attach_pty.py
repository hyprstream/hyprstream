import os, pty, select, signal, sys, time, termios, json, hashlib
from pathlib import Path
binary=sys.argv[1]
master,slave=pty.openpty()
original=termios.tcgetattr(slave)
pid=os.fork()
if pid == 0:
    os.close(master)
    for fd in (0,1,2): os.dup2(slave,fd)
    if slave>2: os.close(slave)
    os.environ['HYPRSTREAM_ATTACH_PTY_CHILD']='1'
    os.execv(binary,[binary,'--exact','cli::tui_handlers::attach_lifecycle_tests::attach_pty_child','--nocapture','--ignored'])
data=b''; sent=False; status=None; deadline=time.monotonic()+10
try:
    while time.monotonic()<deadline:
        ready,_,_=select.select([master],[],[],0.1)
        if ready:
            try: data+=os.read(master,65536)
            except OSError: break
        if not sent and b'Attached to session' in data:
            os.write(master,b'\x02d'); sent=True
        ended,st=os.waitpid(pid,os.WNOHANG)
        if ended: status=st; break
    restored=termios.tcgetattr(slave)==original
    hung=status is None
    if hung:
        os.kill(pid,signal.SIGKILL); _,status=os.waitpid(pid,0)
    result={'binary':binary,'binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),'detach_sent':sent,'loop_returned':b'ATTACH_LOOP_RETURNED' in data,'runtime_dropped':b'ATTACH_RUNTIME_DROPPED' in data,'termios_restored':restored,'hung_after_10s':hung,'wait_status':status,'transcript':data.decode(errors='replace')}
    print(json.dumps(result,indent=2))
    assert sent and restored and result['loop_returned'] and result['runtime_dropped'] and not hung and status == 0, 'attach lifecycle regression'
finally:
    if status is None:
        try: os.kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
        try: os.waitpid(pid,0)
        except ChildProcessError: pass
    os.close(master); os.close(slave)
