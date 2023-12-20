# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 20-12-2023

# --Needed functionalities
# - 1. None

# ---DEPENDENCIES---------------------------------------------------------------


# ---MONITORS-------------------------------------------------------------------
class BestModelCheckpoint:
    def __init__(
        self,
        track="loss",
        mode="min",
        patience=5,
        min_improve=1e-3,
        save_path=None,
        phase="Vl",
    ):
        self.track = track
        self.mode = mode
        self.patience = patience
        self.min_improve = min_improve
        self.save_path = save_path
        self.best = None
        self.best_epoch = None
        self.phase = phase

    def __call__(self, trainer, epoch, phase):
        if phase != self.phase:
            return False, None
        curr = trainer.logs[phase][self.track][-1]
        if self.best is None:
            self.best = curr
            self.best_epoch = epoch
            if self.save_path is not None:
                trainer.save_checkpoint(self.save_path)
        elif (self.mode == "min" and curr < self.best - self.min_improve) or (
            self.mode == "max" and curr > self.best + self.min_improve
        ):
            self.best = curr
            self.best_epoch = epoch
            if self.save_path is not None:
                trainer.save_checkpoint(self.save_path)
        elif epoch - self.best_epoch > self.patience:
            # None for compatibility with other monitors: True to halt
            return True, None
        return False, None


# ---END------------------------------------------------------------------------
