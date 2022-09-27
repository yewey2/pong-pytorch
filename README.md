# Pong PyTorch
Created a DQN network using PyTorch to play pong

To run files, please first install required packages 
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Once installed, you can run training code under `pong_main.py`, or the testing code under `pong_test.py`.

In `pong_test.py`, you can change the file to run on line 17, by inputing the file name. The default file is `pong_best.hdf5`.

To show the game itself, go to `Env.py`, and change `RENDERING = True` (line 3).