#!/usr/bin/env python3

import socket
import argparse
import ase.io
import ase.units
import numpy as np
from copy import deepcopy
from rascal.models.IP_ipi_interface import IPICalculator as IPICalc

description = """
Minimal example of a Python driver connecting to i-PI and exchanging energy, forces, etc.
"""


def recv_data(sock, data):
    """ Fetches binary data from i-PI socket. """
    blen = data.itemsize * data.size
    buf = np.zeros(blen, np.byte)

    bpos = 0
    while bpos < blen:
        timeout = False
        try:
            bpart = 1
            bpart = sock.recv_into(buf[bpos:], blen - bpos)
        except socket.timeout:
            print(" @SOCKET:   Timeout in status recvall, trying again!")
            timeout = True
            pass
        if not timeout and bpart == 0:
            raise RuntimeError("Socket disconnected!")
        bpos += bpart

    if np.isscalar(data):
        return np.frombuffer(buf[0:blen], data.dtype)[0]
    else:
        return np.frombuffer(buf[0:blen], data.dtype).reshape(data.shape)


def send_data(sock, data):
    """ Sends binary data to i-PI socket. """

    if np.isscalar(data):
        data = np.array([data], data.dtype)
    buf = data.tobytes()
    sock.send(buf)


HDRLEN = 12  # number of characters of the default message strings


def Message(mystr):
    """Returns a header of standard length HDRLEN."""

    # convert to bytestream since we'll be sending this over a socket
    return str.ljust(str.upper(mystr), HDRLEN).encode()


def dummy_driver(cell, pos):
    """ Does nothing, but returns properties that can be used by the driver loop."""
    pot = 0.0
    force = pos * 0.0  # makes a zero force with same shape as pos
    vir = cell * 0.0  # makes a zero virial with same shape as cell
    extras = "nada"
    return pot, force, vir, extras


def rascal_driver(cell, icell, pos, model, structure_template):
    """ Computes energies, forces and virials using the IPICalculator class of librascal"""
    Model = IPICalc(model, structure_template)
    cell_tuple = np.zeros((2, 3, 3), float)
    cell_tuple[0] = cell
    cell_tuple[1] = cell
    pot, force, vir = Model.calculate(pos, cell_tuple)
    extras = "nada"
    return pot, force, vir, extras


def driver(
    unix=False, address="", port=12345, model="", struct_templ="", driver=rascal_driver
):
    """Minimal socket client for i-PI."""

    # Opens a socket to i-PI
    if unix:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/ipi_" + address)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raise ValueError("We haven't yet implemented the mf")

    f_init = False
    f_data = False
    f_verbose = False

    # initializes structure arrays
    cell = np.zeros((3, 3), float)
    icell = np.zeros((3, 3), float)
    pos = np.zeros(0, float)

    # initializes return arrays
    pot = 0.0
    force = np.zeros(0, float)
    vir = np.zeros((3, 3), float)
    while True:  # ah the infinite loop!
        header = sock.recv(HDRLEN)

        if header == Message("STATUS"):
            # responds to a status request
            if not f_init:
                sock.sendall(Message("NEEDINIT"))
            elif f_data:
                sock.sendall(Message("HAVEDATA"))
            else:
                sock.sendall(Message("READY"))
        elif header == Message("INIT"):
            # initialization
            rid = recv_data(sock, np.int32())
            initlen = recv_data(sock, np.int32())
            initstr = recv_data(sock, np.chararray(initlen))
            if f_verbose:
                print(rid, initstr)
            f_init = True  # we are initialized now
        elif header == Message("POSDATA"):
            # receives structural information
            cell = recv_data(sock, cell)
            icell = recv_data(
                sock, icell
            )  # inverse of the cell. mostly useless legacy stuff
            nat = recv_data(sock, np.int32())
            if len(pos) == 0:
                # shapes up the position array
                pos.resize((nat, 3))
                force.resize((nat, 3))
            else:
                if len(pos) != nat:
                    raise RuntimeError("Atom number changed during i-PI run")
            pos = recv_data(sock, pos)
            ##### THIS IS THE TIME TO DO SOMETHING WITH THE POSITIONS!
            pot, force, vir, extras = driver(cell, icell, pos, model, struct_templ)
            f_data = True
        elif header == Message("GETFORCE"):
            sock.sendall(Message("FORCEREADY"))

            send_data(sock, np.float64(pot))
            send_data(sock, np.int32(nat))
            send_data(sock, force)
            send_data(sock, vir)
            send_data(sock, np.int32(len(extras)))
            sock.sendall(extras.encode("utf-8"))

            f_data = False
        elif header == Message("EXIT"):
            print("Received exit message from i-PI. Bye bye!")
            return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-u",
        "--unix",
        action="store_true",
        default=False,
        help="Use a UNIX domain socket.",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="localhost",
        help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=12345,
        help="TCP/IP port number. Ignored when using UNIX domain sockets.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="GAP model filename to be used as input to librascal (.json file)",
    )
    parser.add_argument(
        "-s",
        "--struct_templ",
        type=str,
        default="",
        help="Filename for an ASE-compatible Atoms object, used only to initialize atom types and numbers",
    )

    args = parser.parse_args()
    print(args)
    driver(
        unix=args.unix,
        address=args.address,
        port=args.port,
        model=args.model,
        struct_templ=args.struct_templ,
        driver=rascal_driver,
    )
