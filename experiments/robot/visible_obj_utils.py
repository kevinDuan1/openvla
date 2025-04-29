import re
from typing import Dict, List, Tuple

Coord = List[int]  # [x1, y1, x2, y2]
Entry = Tuple[str, Coord]

class StringDictTracker:
    _pattern = re.compile(r'([\w\s]+)\s*\[([\d,\s]+)\]')

    def __init__(self, initial: str = "", iou_threshold: float = 0.5):
        """
        :param initial: optional initial string to parse
        :param iou_threshold: minimum IoU to consider two boxes the same
        """
        self.iou_threshold = iou_threshold
        # flat list of (name, coords)
        self._entries: List[Entry] = []
        self._string = ""
        if initial:
            self.update(initial)

    @staticmethod
    def _iou(boxA: Coord, boxB: Coord) -> float:
        # compute intersection
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        # compute union
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = areaA + areaB - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def update(self, s: str) -> None:
        """
        Parse new boxes from `s` and for each:
        - if it overlaps an existing entry by >= iou_threshold, replace that entry
        - otherwise append as a new entry
        Then rebuild the internal string.
        """
        for match in self._pattern.finditer(s):
            key_new = match.group(1).strip()
            coords_new = [int(n) for n in match.group(2).split(',')]
            replaced = False

            # try to match with existing entries
            for idx, (key_old, coords_old) in enumerate(self._entries):
                if self._iou(coords_old, coords_new) >= self.iou_threshold:
                    # replace old entry
                    self._entries[idx] = (key_new, coords_new)
                    replaced = True
                    break

            if not replaced:
                self._entries.append((key_new, coords_new))

        # rebuild the string representation
        self._string = ", ".join(
            f"{name} [{', '.join(map(str, box))}]" 
            for name, box in self._entries
        )

    @property
    def current_string(self) -> str:
        """The up-to-date reconstructed string."""
        return self._string

    @property
    def current_dict(self) -> Dict[str, List[Coord]]:
        """
        Groups entries by name. 
        Returns a dict mapping each name to either a single list or list of lists.
        """
        d: Dict[str, List[Coord]] = {}
        for name, box in self._entries:
            d.setdefault(name, []).append(box)
        # flatten singletons
        return {k: v if len(v) > 1 else v[0] for k, v in d.items()}

    def __str__(self) -> str:
        return self.current_string