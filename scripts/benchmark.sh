parallel -j1 python benchmark.py --exp-name APL_MNIST_std_seeded --dataset MNIST --high-perf 0 --port 0 --mlflow_port 8082 --mode {} --target_value {} ::: 0 1 2 ::: 0 0.5 1.0 2.0 4.0 8.0
parallel -j1 python benchmark.py --exp-name APL_MS_std_seeded --dataset MS --high-perf 0 --port 0 --mlflow_port 8082 --mode 1 --target_value {} ::: 0 0.5 1.0 2.0 4.0 8.0
parallel -j1 python benchmark.py --exp-name APL_SC_std_seeded --dataset SC --high-perf 0 --port 0 --mlflow_port 8082 --mode 1 --target_value {} ::: 0 0.5 1.0 2.0 4.0 8.0
parallel -j1 python benchmark.py --exp-name APL_MNIST_std_seeded --dataset MNIST --high-perf 0 --port 0 --mlflow_port 8082 --mode 2 --target_value {} ::: 0 0.5 1.0 2.0 4.0 8.0
parallel -j1 python benchmark.py --exp-name APL_MS_std_seeded --dataset MS --high-perf 0 --port 0 --mlflow_port 8082 --mode 2 --target_value {} ::: 0 0.5 1.0 2.0 4.0 8.0
parallel -j1 python benchmark.py --exp-name APL_SC_std_seeded --dataset SC --high-perf 0 --port 0 --mlflow_port 8082 --mode 2 --target_value {} ::: 0 0.5 1.0 2.0 4.0 8.0
