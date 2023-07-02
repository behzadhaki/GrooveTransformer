from eval.UMAP import UMapper

if __name__ == "__main__":
    import numpy as np
    umapper = UMapper("test")
    umapper.fit(np.random.random((1024, 128)), tags_=np.random.choice(["a", "b", "c", "d", "e", "f", "g", "h", "i", "m"], 1024).tolist())
    umapper.plot(show_plot=True)
    p = umapper.plot(show_plot=False, prepare_for_wandb=True)