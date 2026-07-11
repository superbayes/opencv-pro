#pragma once

#include "contour_tracker.h"

namespace liquid {

void smoothContours(ContourTrack& track, int windowRadius = 4);

}  // namespace liquid
