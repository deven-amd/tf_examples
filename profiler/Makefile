SHELL=/bin/bash

example_00:
	rm -rf timeline_00.json
	rm -rf run_00.log
	HIP_TRACE_API=1 LOG_LEVEL=3 python3 example_00.py &> run_00.log

example_01:
	rm -rf timeline_01.json
	rm -rf run_01.log
	HIP_TRACE_API=1 LOG_LEVEL=3 python3 example_01.py &> run_01.log

example_02:
	rm -rf logs_02
	python3 example_02.py


clean:
	rm -rf *.log *.json logs_*
