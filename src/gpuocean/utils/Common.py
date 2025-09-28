# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the different helper functions and 
classes that are shared through out all elements of the code.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations
import os
import time
import signal
import subprocess
import tempfile
import sys
import logging
import gc
import warnings
import functools
from enum import IntEnum
from dataclasses import dataclass

import numpy as np


class PopenFileBuffer(object):
    """
    Simple class for holding a set of tempfiles
    for communicating with a subprocess
    """

    def __init__(self):
        self.stdout = tempfile.TemporaryFile(mode='w+t')
        self.stderr = tempfile.TemporaryFile(mode='w+t')

    def __del__(self):
        self.stdout.close()
        self.stderr.close()

    def read(self):
        self.stdout.seek(0)
        cout = self.stdout.read()
        self.stdout.seek(0, 2)

        self.stderr.seek(0)
        cerr = self.stderr.read()
        self.stderr.seek(0, 2)

        return cout, cerr


class IPEngine(object):
    """
    Class for starting IPEngines for MPI processing in IPython
    """

    def __init__(self, n_engines):
        self.logger = logging.getLogger(__name__)

        # Start ipcontroller
        self.logger.info("Starting IPController")
        self.c_buff = PopenFileBuffer()
        c_cmd = ["ipcontroller", "--ip='*'"]
        c_params = dict()
        c_params['stderr'] = self.c_buff.stderr
        c_params['stdout'] = self.c_buff.stdout
        c_params['shell'] = False
        if os.name == 'nt':
            c_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.c = subprocess.Popen(c_cmd, **c_params)

        # Wait until controller is running
        time.sleep(3)

        # Start engines
        self.logger.info("Starting IPEngines")
        self.e_buff = PopenFileBuffer()
        e_cmd = ["mpiexec", "-n", str(n_engines), "ipengine", "--mpi"]
        e_params = dict()
        e_params['stderr'] = self.e_buff.stderr
        e_params['stdout'] = self.e_buff.stdout
        e_params['shell'] = False
        if os.name == 'nt':
            e_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.e = subprocess.Popen(e_cmd, **e_params)

        # attach to a running cluster
        import ipyparallel
        self.cluster = ipyparallel.Client()  # profile='mpi')
        time.sleep(3)
        while len(self.cluster.ids) != n_engines:
            time.sleep(0.5)
            self.logger.info("Waiting for cluster...")
            self.cluster = ipyparallel.Client()  # profile='mpi')

        self.logger.info("Done")

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.e is not None:
            if os.name == 'nt':
                self.logger.warning("Sending CTRL+C to IPEngine")
                self.e.send_signal(signal.CTRL_C_EVENT)

            try:
                self.e.communicate(timeout=3)
                self.e.kill()
            except subprocess.TimeoutExpired:
                self.logger.warning("Killing IPEngine")
                self.e.kill()
                self.e.communicate()
            self.e = None

            cout, cerr = self.e_buff.read()
            self.logger.info("IPEngine cout: {:s}".format(cout))
            self.logger.info("IPEngine cerr: {:s}".format(cerr))
            self.e_buff = None

            gc.collect()

        if self.c is not None:
            if os.name == 'nt':
                self.logger.warning("Sending CTRL+C to IPController")
                self.c.send_signal(signal.CTRL_C_EVENT)

            try:
                self.c.communicate(timeout=3)
                self.c.kill()
            except subprocess.TimeoutExpired:
                self.logger.warning("Killing IPController")
                self.c.kill()
                self.c.communicate()
            self.c = None

            cout, cerr = self.c_buff.read()
            self.logger.info("IPController cout: {:s}".format(cout))
            self.logger.info("IPController cerr: {:s}".format(cerr))
            self.c_buff = None

            gc.collect()

    def elapsed(self):
        return time.time() - self.start


class ProgressPrinter(object):
    """
    Small helper class for 
    """

    def __init__(self, print_every=5):
        self.logger = logging.getLogger(__name__)
        self.start = time.time()
        self.print_every = print_every
        self.next_print_time = print_every
        self.print_string = ProgressPrinter.formatString(0, 0, 0)
        self.last_x = 0
        self.secs_per_iter = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def getPrintString(self, x: int) -> str | None:
        elapsed = time.time() - self.start

        if x == 0:
            self.print_string = ProgressPrinter.formatString(0, 0, np.nan)
            return self.print_string

        if elapsed >= self.next_print_time or x == 1.0:
            dt = elapsed - (self.next_print_time - self.print_every)
            dx = x - self.last_x

            if dt <= 0:
                return None

            self.last_x = x
            self.next_print_time = max(elapsed, self.next_print_time + self.print_every)

            # A kind of floating average
            if not self.secs_per_iter:
                self.secs_per_iter = dt / dx
            self.secs_per_iter = 0.2 * self.secs_per_iter + 0.8 * (dt / dx)

            remaining_time = (1 - x) * self.secs_per_iter

            self.print_string = ProgressPrinter.formatString(x, elapsed, remaining_time)

        return self.print_string

    @staticmethod
    def formatString(t: float, elapsed: float, remaining_time: float):
        return (f"{ProgressPrinter.progressBar(t)}. "
                f"Total: {ProgressPrinter.timeString(elapsed + remaining_time)}, "
                f"elapsed: {ProgressPrinter.timeString(elapsed)}, "
                f"remaining: {ProgressPrinter.timeString(remaining_time)}")

    @staticmethod
    def timeString(seconds: float):
        if np.isnan(seconds):
            return str(seconds)

        seconds = int(max(seconds, 0))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        periods = [('h', hours), ('m', minutes), ('s', seconds + sys.float_info.epsilon)]
        time_string = ' '.join('{:d}{:s}'.format(int(value), name)
                               for name, value in periods
                               if value)
        return time_string

    @staticmethod
    def progressBar(t: float, width=30):
        progress = int(round(width * t))
        progressbar = "0% [" + "#" * progress + "=" * (width - progress) + "] 100%"
        return progressbar


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated.
    Reference: https://stackoverflow.com/questions/2536307/how-do-i-deprecate-python-functions
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # MLS: Seems wrong to mess with standard filter settings in this context.
        # Code is nevertheless not removed, as this could be relevant at a later time.
        # warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(f"Call to deprecated function {func.__name__}.",
                      category=DeprecationWarning,
                      stacklevel=2)
        # warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func

class BoundaryType(IntEnum):
    WALL = 1
    PERIODIC = 2
    FLOW_RELAXATION_SCHEME = 3
    OPEN_LINEAR_INTERPOLATION = 4

    def __str__(self):
        return self.name.title()

@dataclass
class SpongeCells:
    north: int
    east: int
    south: int
    west: int

    def __str__(self):
        return f"SpongeCells: {{north: {self.north}, east: {self.east}, south: {self.south}, west: {self.west}}}"


class BoundaryConditions:
    """
    Class that represents different forms for boundary conditions
    """

    def __init__(self,
                 north: BoundaryType=BoundaryType.WALL, east: BoundaryType=BoundaryType.WALL,
                 south: BoundaryType=BoundaryType.WALL, west: BoundaryType=BoundaryType.WALL,
                 sponge_cells: SpongeCells = None):
        """
        There is one parameter for each of the cartesian boundaries.
        Values can be set as follows:
        1 = Wall
        2 = Periodic (requires same for opposite boundary as well)
        3 = Open Boundary with Flow Relaxation Scheme
        4 = Open linear interpolation
        Options 3 and 4 are of sponge type (requiring extra computational domain)
        """
        if sponge_cells is None:
            sponge_cells = SpongeCells(0, 0, 0, 0)
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.spongeCells = sponge_cells

        # Checking that periodic boundaries are periodic
        assert not ((self.north == BoundaryType.PERIODIC or self.south == BoundaryType.PERIODIC) and
                    (self.north != self.south)), \
            'The given periodic boundary conditions are not periodically (north/south)'
        assert not ((self.east == BoundaryType.PERIODIC or self.west == BoundaryType.PERIODIC) and
                    (self.east != self.west)), \
            'The given periodic boundary conditions are not periodically (east/west)'

    def get(self):
        raise NotImplementedError("get() implementation changed! Change calling code")
        # Previous code was:
        # return [self.north, self.east, self.south, self.west]
        # New code is the following, change code calling this function to support new dict
        return {'north': self.north, 'south': self.south, 'east': self.east, 'west': self.west}

    def getBCDict(self):
        return {'north': self.north, 'south': self.south, 'east': self.east, 'west': self.west}

    def getSponge(self):
        return self.spongeCells

    def isDefault(self):
        return (self.north == BoundaryType.WALL and
                self.east == BoundaryType.WALL and
                self.south == BoundaryType.WALL and
                self.east == BoundaryType.WALL)

    def isSponge(self):
        return (self.north == BoundaryType.FLOW_RELAXATION_SCHEME or self.north == BoundaryType.OPEN_LINEAR_INTERPOLATION or
                self.east == BoundaryType.FLOW_RELAXATION_SCHEME or self.east == BoundaryType.OPEN_LINEAR_INTERPOLATION or
                self.south == BoundaryType.FLOW_RELAXATION_SCHEME or self.south == BoundaryType.OPEN_LINEAR_INTERPOLATION or
                self.west == BoundaryType.FLOW_RELAXATION_SCHEME or self.west == BoundaryType.OPEN_LINEAR_INTERPOLATION)

    def isPeriodicNorthSouth(self):
        return self.north == BoundaryType.PERIODIC and self.south == BoundaryType.PERIODIC

    def isPeriodicEastWest(self):
        return self.east == BoundaryType.PERIODIC and self.west == BoundaryType.PERIODIC

    def isPeriodic(self):
        return self.isPeriodicEastWest() and self.isPeriodicNorthSouth()

    def __str__(self):
        msg = f"north: {self.north}, east: {self.east}, south: {self.south}, west: {self.west}, {self.spongeCells}"
        return msg

    @classmethod
    def fromstring(cls, bc_string: str) -> BoundaryConditions:

        def keyword_to_cond(key: str) -> BoundaryType | None:
            if key == str(BoundaryType.WALL):
                return BoundaryType.WALL
            elif key == str(BoundaryType.PERIODIC):
                return BoundaryType.PERIODIC
            elif key == str(BoundaryType.FLOW_RELAXATION_SCHEME):
                return BoundaryType.FLOW_RELAXATION_SCHEME
            elif key == str(BoundaryType.OPEN_LINEAR_INTERPOLATION):
                return BoundaryType.OPEN_LINEAR_INTERPOLATION
            else:
                return None

        # clean string
        bc_clean_str = bc_string.replace(',', '').replace('}', '').replace('{', '').replace('[', '').replace(']',
                                                                                                             '').replace(
            ':', '').replace("'", "")

        bc_array = str.split(bc_clean_str, ' ')
        north = keyword_to_cond(bc_array[1])
        east = keyword_to_cond(bc_array[3])
        south = keyword_to_cond(bc_array[5])
        west = keyword_to_cond(bc_array[7])

        if len(bc_array) > 13:
            sponge_cells = {bc_array[9]: int(bc_array[10]),
                            bc_array[11]: int(bc_array[12]),
                            bc_array[13]: int(bc_array[14]),
                            bc_array[15]: int(bc_array[16])}
            sponge_cells = SpongeCells(**sponge_cells)
        else:
            sponge_cells = SpongeCells(int(bc_array[9]),
                            int(bc_array[10]),
                            int(bc_array[11]),
                            int(bc_array[12]))

        return cls(north=north, east=east, south=south, west=west, sponge_cells=sponge_cells)


class SingleBoundaryConditionData:
    """
    This class holds the external solution for a single boundary over time.
    """

    def __init__(self, h=None, hu=None, hv=None):
        self.h = [np.zeros(2, dtype=np.float32, order='C')]
        self.hu = [np.zeros(2, dtype=np.float32, order='C')]
        self.hv = [np.zeros(2, dtype=np.float32, order='C')]
        self.numSteps = 1
        self.shape = self.h[0].shape

        if h is not None:
            self.shape = h[0].shape
            self.numSteps = len(h)

            self.h = h
            self.hu = hu
            self.hv = hv

            for i in range(len(h)):
                assert (h[i].shape == self.shape), str(self.shape) + " vs " + str(h[i].shape)
                assert (hu[i].shape == self.shape), str(self.shape) + " vs " + str(hu[i].shape)
                assert (hv[i].shape == self.shape), str(self.shape) + " vs " + str(hv[i].shape)

                assert (h[i].dtype == 'float32'), "h data needs to be of type np.float32"
                assert (hu[i].dtype == 'float32'), "hu data needs to be of type np.float32"
                assert (hv[i].dtype == 'float32'), "hv data needs to be of type np.float32"

    def __str__(self):
        return str(self.numSteps) + " steps, each " + str(self.shape)


class BoundaryConditionsData:
    """
    This class holds external solution for all boundaries over time.
    """

    def __init__(self,
                 t: list[float] = None,
                 north=SingleBoundaryConditionData(),
                 south=SingleBoundaryConditionData(),
                 east=SingleBoundaryConditionData(),
                 west=SingleBoundaryConditionData()):

        self.t: list[float] = [0]
        self.numSteps = 1
        self.north = north
        self.south = south
        self.east = east
        self.west = west

        if t is not None:
            self.t = t
            self.numSteps = len(t)

        for data in [north, south, east, west]:
            assert (data.numSteps == self.numSteps), "Wrong timesteps " + str(data.numSteps) + " vs " + str(
                self.numSteps)

        assert (north.h[0].shape == south.h[0].shape), "Wrong shape of north vs south " + str(
            north.h[0].shape) + " vs " + str(south.h[0].shape)
        assert (east.h[0].shape == west.h[0].shape), "Wrong shape of east vs west " + str(
            east.h[0].shape) + " vs " + str(west.h[0].shape)

    def __str__(self):
        return "Steps=" + str(self.numSteps) \
            + ", [north=" + str(self.north) + "]" \
            + ", [south=" + str(self.south) + "]" \
            + ", [east=" + str(self.east) + "]" \
            + ", [west=" + str(self.west) + "]"
