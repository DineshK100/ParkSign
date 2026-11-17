# Result of classifying one sign post.
std_msgs/Header header
string spot_id
string sign_class               # one of the five classes, or "unknown"
string text_read
float32 confidence
string backend                  # "claude" | "ocr" | "ground_truth"
string ground_truth_class       # filled in simulation for evaluation
float32 latency_s
