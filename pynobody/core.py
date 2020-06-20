#!/usr/bin/env python

__all__ = ['Body', 'Orbit']

from random import getrandbits
from collections import defaultdict

from progressbar import ProgressBar

import numpy as np


class Body(object):
    """
    A Body
    """

    def __init__(self, name=None, position=None, velocity=None, mass=None):
        if not isinstance(name, (type(None), str)):
            raise TypeError(
                "'name' should be one of 'str' or 'None' object.")

        if not isinstance(position, (type(None), list)):
            if not (type(position).__module__ == np.__name__):
                raise TypeError(
                    "'position' should be one of 'None', 'list' or 'np.array'.")

        if not isinstance(velocity, (type(None), list)):
            if not (type(velocity).__module__ == np.__name__):
                raise TypeError(
                    "'velocity' should be one of 'None', 'list' or 'np.array'.")

        if not isinstance(mass, (int, float)):
            raise TypeError(
                "'mass' should be one of 'int' of 'float'.")

        if name is None:
            name = str(getrandbits(16))

        self.name = name

        if position is None:
            self.position = np.array([0., 0., 0.], dtype=np.float64)
        elif isinstance(position, list):
            self.position = np.array(position, dtype=np.float64)
        else:
            self.position = position

        if velocity is None:
            self.velocity = np.array([0., 0., 0.], dtype=np.float64)
        elif isinstance(position, list):
            self.velocity = np.array(position, dtype=np.float64)
        else:
            self.velocity = velocity

        if mass is None:
            mass = 1.

        self.mass = mass


class Orbit(object):

    def __init__(self, dt_param=1e-3, dt_dia=1, dt_out=1, dt_tot=10,
                 init_out=False, dimension=3, dtype=np.float64,
                 stdout=False):

        self.dt_param = dt_param
        self.dt_dia = dt_dia
        self.dt_out = dt_out
        self.dt_tot = dt_tot

        self.init_out = init_out
        self.init_flag = True
        self.stdout = stdout

        self.ndim = dimension
        self.dtype = dtype
        self.n = 0
        self.t = 0

        self.bodies = list()

        self.name = list()
        self.mass = None
        self.pos = None
        self.vel = None
        self.acc = None
        self.jerk = None

        self.old_pos = None
        self.old_vel = None
        self.old_acc = None
        self.old_jerk = None

        self.einit = 0
        self.epot = 0
        self.coll_time = 0
        self.nsteps = 0
        self.dt = 0

        self.states = defaultdict(list)
        self.erg = list()

        self.bar = None

    def add_body(self, body):
        if not isinstance(body, (list, Body)):
            raise TypeError("'body' should be 'list' or 'Body' object.")

        if isinstance(body, Body):
            body = [body]

        self.bodies += body

    def prepare(self):
        self.n = len(self.bodies)

        self.mass = np.zeros((self.n, 1), dtype=self.dtype)
        self.pos = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.vel = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.acc = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.jerk = np.zeros((self.n, self.ndim), dtype=self.dtype)

        self.old_pos = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.old_vel = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.old_acc = np.zeros((self.n, self.ndim), dtype=self.dtype)
        self.old_jerk = np.zeros((self.n, self.ndim), dtype=self.dtype)

        for i, body in enumerate(self.bodies):
            self.name.append(body.name)
            self.mass[i] = body.mass
            self.pos[i, :] = body.position
            self.vel[i, :] = body.velocity

            self.states[body.name] = list()

        self.bar = ProgressBar(max_value=int(self.dt_tot / self.dt_param))

    def run(self):
        self.prepare()
        self.evolve()

    def evolve(self):
        msg = f"Starting a Hermite integration for a {self.n}-body " \
              f"system,\n  from time t = {self.t} " \
              f"with time step control parameter dt_param = {self.dt_param}" \
              f" until time {self.t + self.dt_tot},\n" \
              f"  with diagnostics output interval dt_dia = {self.dt_dia},\n" \
              f"  and snapshot output interval dt_out = {self.dt_out}.\n"

        if self.stdout:
            print(msg)

        self.get_acc_jerk_pot_coll()

        self.write_diagnostics()
        self.init_flag = False

        if self.init_out:
            self.put_snapshot()

        t_dia = self.t + self.dt_dia
        t_out = self.t + self.dt_out
        t_end = self.t + self.dt_tot

        c = 0
        while True:
            while (self.t < t_dia) and (self.t < t_out) and (self.t < t_end):
                self.dt = self.dt_param * self.coll_time
                self.evolve_step()
                self.nsteps += 1

            if self.t >= t_dia:
                self.write_diagnostics()
                t_dia += self.dt_dia

            if self.t >= t_out:
                self.put_snapshot()
                t_out += self.dt_out
                self.bar.update(c)
                c += 1

            if self.t >= t_end:
                break

        self.bar.finish()
        self.finish()

    def get_acc_jerk_pot_coll(self):
        self.epot = 0
        coll_time_q = 1e300

        self.acc *= 0
        self.jerk *= 0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                rji = self.pos[j, :] - self.pos[i, :]
                vji = self.vel[j, :] - self.vel[i, :]

                r2 = (rji ** 2).sum()
                v2 = (vji ** 2).sum()
                rv_r2 = (rji * vji).sum()

                rv_r2 /= r2
                r = np.sqrt(r2)
                r3 = r * r2

                self.epot -= self.mass[i][0] * self.mass[j][0] / r

                da = rji / r3
                dj = (vji - 3 * rv_r2 * rji) / r3

                self.acc[i, :] += self.mass[j][0] * da
                self.acc[j, :] -= self.mass[i][0] * da
                self.jerk[i, :] += self.mass[j][0] * dj
                self.jerk[j, :] -= self.mass[i][0] * dj

                coll_est_q = (r2 ** 2) / (v2 ** 2)
                if coll_time_q > coll_est_q:
                    coll_time_q = coll_est_q

                da2 = (da**2).sum()

                mij = self.mass[i][0] + self.mass[j][0]
                da2 *= mij ** 2

                coll_est_q = r2 / da2
                if coll_time_q > coll_est_q:
                    coll_time_q = coll_est_q

        self.coll_time = np.sqrt(np.sqrt(coll_time_q))

    def write_diagnostics(self):
        ekin = (0.5 * self.mass * self.vel ** 2).sum()
        etot = ekin + self.epot

        if self.init_flag:
            self.einit = etot

        e_abs = etot - self.einit
        e_rel = e_abs / self.einit

        message = f"at time t = {self.t}, after {self.nsteps} steps:\n" \
                  f"  E_kin = {ekin}, E_pot = {self.epot}, E_tot = {etot}\n" \
                  f"  absolute energy error: E_tot - E_init = " \
                  f"{e_abs}\n" \
                  f"  relative energy error: (E_tot - E_init / E_init = " \
                  f"{e_rel}\n"

        if self.stdout:
            print(message)

        row = [self.t, self.nsteps, ekin, self.epot, etot, e_abs, e_rel]
        self.erg.append(row)

    def put_snapshot(self):
        if self.stdout:
            print(self.n)
            print(self.t)

        for i in range(self.n):
            if self.stdout:
                row = np.array2string(self.mass[i][0])
                row += np.array2string(self.pos[i, :])
                row += np.array2string(self.vel[i, :])

                print(row)

            row = [p for p in self.pos[i]]
            row += [v for v in self.vel[i]]
            self.states[self.bodies[i].name].append(row)

    def evolve_step(self):
        self.old_pos[:, :] = self.pos[:, :]
        self.old_vel[:, :] = self.vel[:, :]
        self.old_acc[:, :] = self.acc[:, :]
        self.old_jerk[:, :] = self.jerk[:, :]

        self.predict_step()
        self.get_acc_jerk_pot_coll()
        self.correct_step()

        self.t += self.dt

    def predict_step(self):
        self.pos += self.vel * self.dt + \
                    ((self.acc * self.dt ** 2) / 2.) + \
                    ((self.jerk * self.dt ** 3) / 6.)

        self.vel += self.acc * self.dt + ((self.jerk * self.dt ** 2) / 2.)

    def correct_step(self):
        self.vel = self.old_vel + \
            (self.old_acc + self.acc) * self.dt / 2. + \
            (self.old_jerk - self.jerk) * self.dt * self.dt / 12.

        self.pos = self.old_pos + \
            (self.old_vel + self.vel) * self.dt / 2. + \
            (self.old_acc - self.acc) * self.dt * self.dt / 12.

    def finish(self):
        self.erg = np.array(self.erg, dtype=self.dtype)

        for body in self.bodies:
            self.states[body.name] = \
                np.array(self.states[body.name], dtype=self.dtype)

    def get_energy(self, key):
        db = {'t': 0, 'step': 1, 'ekin': 2, 'epot': 3,
              'etot': 4, 'e_abs': 5, 'e_rel': 6}

        try:
            return self.erg[:, db[key]]
        except IndexError:
            return None

    def get_state(self, body):
        state = dict()

        state['x'] = self.states[body.name][:, 0]
        state['y'] = self.states[body.name][:, 1]
        state['z'] = self.states[body.name][:, 2]
        state['vx'] = self.states[body.name][:, 3]
        state['vy'] = self.states[body.name][:, 4]
        state['vz'] = self.states[body.name][:, 5]

        return state
