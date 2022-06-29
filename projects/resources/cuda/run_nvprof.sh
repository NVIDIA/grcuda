/usr/local/cuda/bin/nsys nvprof --csv -o b1m --profile-from-start off bin/b -m 8 -d -v -t 30 -k b1m -n 950000000 -b 32 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b5m --profile-from-start off bin/b -m 8 -d -v -t 30 -k b5m -n 35000000 -b 1024 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b6m --profile-from-start off bin/b -m 8 -d -v -t 30 -k b6m -n 1800000 -b 32 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b9m --profile-from-start off bin/b -m 8 -d -v -t 30 -k b9m -n 60000 -b 32 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b6m_4 --profile-from-start off bin/b -m 4 -d -v -t 30 -k b6m -n 1800000 -b 32 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b9m_4 --profile-from-start off bin/b -m 4 -d -v -t 30 -k b9m -n 60000 -b 32 -g 64
/usr/local/cuda/bin/nsys nvprof --csv -o b11m --profile-from-start off bin/b -m 8 -d -v -t 30 -k b11m -n 60000 -b 256 -g 64

/usr/local/cuda/bin/nsys stats --report gputrace --format csv b1m.sqlite  -o b1m
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b5m.sqlite  -o b5m
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b6m.sqlite  -o b6m
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b9m.sqlite  -o b9m
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b11m.sqlite  -o b11m
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b6m_4.sqlite  -o b6m_4
/usr/local/cuda/bin/nsys stats --report gputrace --format csv b9m_4.sqlite  -o b9m_4

