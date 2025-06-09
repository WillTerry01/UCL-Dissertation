function theta = normalize_thetas(theta)

%return
theta = atan2(sin(theta), cos(theta));
