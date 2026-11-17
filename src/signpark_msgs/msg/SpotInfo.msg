# One parking spot in the lot map.
string id                       # e.g. "P03"
string geometry                 # "perpendicular" | "parallel" | "angled"
geometry_msgs/Pose2D center     # spot center (map frame); theta = "into the spot" heading
float32 width
float32 length
string sign_class               # "head_in" | "back_in" | "parallel" | "angled" | "no_parking" | "unknown"
string sign_text                # text the reader extracted (may be empty)
float32 sign_confidence
bool occupied
float32 dist_to_entrance        # meters
