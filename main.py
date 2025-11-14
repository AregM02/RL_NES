from tcp_conncection.environment import FceuxEnv
import time

env = FceuxEnv()

t0 = time.time()
while True:
    frame = env.step()

    if frame is None:
        env.close()
        break
    
    time.sleep(0.01)
    print(f"{time.time()-t0:.2f}::{frame.shape}")