import numpy as np

from .types import State, Point3D

class Target:
    def __init__(self, track):
        self.track_time = track[:, 0]
        self.track_y = track[:, 1]
        self.track_x = track[:, 2]
        self.track_z = np.zeros_like(self.track_x)

    def get_time_bounds(self):
        return self.track_time[5], self.track_time[-6]

    def get_state(self, time) -> State:
        idx = np.searchsorted(self.track_time, time, side='right') - 1

        # state is not established in model
        t0 = self.track_time[idx-1]
        t1 = self.track_time[idx]
        t2 = self.track_time[idx+1]

        x0 = self.track_x[idx-1]
        x1 = self.track_x[idx]
        x2 = self.track_x[idx+1]

        y0 = self.track_y[idx-1]
        y1 = self.track_y[idx]
        y2 = self.track_y[idx+1]

        d0 = (t0-t1)*(t0-t2)
        d1 = (t1-t0)*(t1-t2)
        d2 = (t2-t0)*(t2-t1)

        L0 = ((time-t1)*(time-t2)) / d0
        L1 = ((time-t0)*(time-t2)) / d1
        L2 = ((time-t0)*(time-t1)) / d2

        # position
        _x = x0*L0 + x1*L1 + x2*L2
        _y = y0*L0 + y1*L1 + y2*L2

        # first derivatives
        L0p = ((time-t2) + (time-t1)) / d0
        L1p = ((time-t2) + (time-t0)) / d1
        L2p = ((time-t1) + (time-t0)) / d2

        _vx = x0*L0p + x1*L1p + x2*L2p
        _vy = y0*L0p + y1*L1p + y2*L2p

        # second derivatives (constant)
        L0pp = 2 / d0
        L1pp = 2 / d1
        L2pp = 2 / d2

        _ax = x0*L0pp + x1*L1pp + x2*L2pp
        _ay = y0*L0pp + y1*L1pp + y2*L2pp

        return State(
            Point3D(_x, _y, 0),
            Point3D(_vx, _vy, 0),
            Point3D(_ax, _ay, 0),
            time
        )