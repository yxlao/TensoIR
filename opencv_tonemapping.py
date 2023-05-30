from pathlib import Path
import cv2
import numpy as np
import camtools as ct


def main():
    hdr_path = Path.home() / "research/object-relighting-dataset/dataset/antman/test/gt_env_512_rotated_0000.hdr"

    # Must read with cv2.IMREAD_ANYDEPTH
    hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH)

    # Apply -6 EV exposure compensation
    hdr  = hdr / (2** 6.0)

    # Tonemap
    ldr = hdr ** (1 / 2.2)

    # Clip to [0, 1], this can be done here or after exposure compensation
    ldr = np.clip(ldr, 0, 1)
    
    # BGR to RGB
    ldr = ldr[:, :, ::-1]
    
    # Save
    ct.io.imwrite("ldr.png", ldr)


if __name__ == "__main__":
    main()
