#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin


def main():
    step = "<STEP>"
    for line in stdin:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split(" ")
        inp = []
        out = []
        pairs = []
        for pair in parts:
            if pair.startswith("::"):
                src = ":"
                tgt = pair[2:]
            else:
                src, tgt = pair.split(":", maxsplit=1)
            pairs.append((src, tgt))
        for i, (src, tgt) in enumerate(pairs):
            if src:
                inp.append(src)
            if tgt:
                out.append(tgt)
            if i < len(pairs) - 1 and pairs[i+1][0] != "":
                out.append(step)
        print("{}\t{}".format(" ".join(inp), " ".join(out)))


if __name__ == '__main__':
    main()
