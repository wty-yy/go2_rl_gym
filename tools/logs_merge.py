import time
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from tensorboard.compat.proto import event_pb2
from tensorboard.backend.event_processing import event_file_loader

PATH_PARENT = Path(__file__).parent.resolve()
BASE_COLUMNS = [
    'it', 'benchmark',
    'lin_vel_err',
    'ang_vel_err',
    'dof_limits',
    'dof_power',
    'orientation_stability',
    'torque_smoothness',
    'flat', 'wave', 'obstacle',
    'slope_fd', 'slope_bd',
    'stairs_fd', 'stairs_bd',
    'terrain_level'
]

def fast_read(event_file_path, tag_names):
    loader = event_file_loader.RawEventFileLoader(event_file_path)
    steps = []
    values = []

    for raw_event in loader.Load():
        event = event_pb2.Event.FromString(raw_event)
        
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.tag in tag_names:
                    steps.append(event.step)
                    values.append(value.simple_value)
                    
    return pd.DataFrame({'step': steps, 'value': values})

class Collector:
    def __init__(self, log_dirs):
        self.log_dirs = Path(log_dirs)
        assert self.log_dirs.exists(), f"Log directory {log_dirs} does not exist."
        assert self.log_dirs.is_dir(), f"{log_dirs} is not a directory."
        alg_name = self.log_dirs.parent.name
        date_str = self.log_dirs.name
        self.output_dir = PATH_PARENT / f"{alg_name}_{date_str}"
        if self.output_dir.exists():
            s = input(f"[Warning] Output directory {self.output_dir} already exists, press Enter to continue and overwrite or type 'q' to quit...")
            if s.lower() == 'q':
                exit(0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv = self.output_dir / f"{alg_name}_{date_str}.csv"
        self.datas = defaultdict(list)


        self.output_tb = self.output_dir / "tb.csv"
        if self.output_tb.exists():
            print(f"Loading existing tensorboard data from {self.output_tb}")
            self.tb_df = pd.read_csv(self.output_tb)
        else:
            start_time = time.time()
            print(f"Start reading tensorboard events at {time.ctime(start_time)}")
            self.tb_df = fast_read(str(self.log_dirs.glob("events.out.tfevents.*").__next__()), ['Terrain/terrain_level_all', 'Episode/terrain_level_all'])
            print(f"Finished reading tensorboard events in {time.time() - start_time:.2f} seconds.")
            self.tb_df.to_csv(self.output_tb, index=False)
            print(f"Saved tensorboard data to {self.output_tb}")
    
    def collect(self):
        robogauge_results_path = self.log_dirs / "robogauge_results"
        results = list(robogauge_results_path.glob("*.yaml"))
        results = sorted(results, key=lambda x: int(x.stem.split("_")[-1]))
        for result in tqdm(results):
            it = int(result.stem.split("_")[-1])
            with open(result, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.datas['it'].append(it)
            self.datas['benchmark'].append(float(data['benchmark_score']))
            for metric_name in [
                'lin_vel_err',
                'ang_vel_err',
                'dof_limits',
                'dof_power',
                'orientation_stability',
                'torque_smoothness'
            ]:
                self.datas[f'{metric_name}_mean'].append(float(data['summary'][metric_name]['mean'].split(' ')[0]))
                self.datas[f'{metric_name}_mean@25'].append(float(data['summary'][metric_name]['mean@25'].split(' ')[0]))
                self.datas[f'{metric_name}_mean@50'].append(float(data['summary'][metric_name]['mean@50'].split(' ')[0]))
            for terrain_name in [
                'flat',
                'wave',
                'obstacle',
                'slope_fd',
                'slope_bd',
                'stairs_fd',
                'stairs_bd',
            ]:
                if data['robust_score'][terrain_name] is None:
                    self.datas[f'{terrain_name}_mean'].append(0.0)
                    self.datas[f'{terrain_name}_mean@25'].append(0.0)
                    self.datas[f'{terrain_name}_mean@50'].append(0.0)
                    continue
                self.datas[f'{terrain_name}_mean'].append(float(data['robust_score'][terrain_name]['mean']))
                self.datas[f'{terrain_name}_mean@25'].append(float(data['robust_score'][terrain_name]['mean@25']))
                self.datas[f'{terrain_name}_mean@50'].append(float(data['robust_score'][terrain_name]['mean@50']))
            
            self.datas['terrain_level'].append(float(self.tb_df[self.tb_df['step'] == it]['value'].values[0]))
        df = pd.DataFrame(self.datas)
        df.to_csv(self.output_csv, index=False)
        print(f"Saved merged results to {self.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dirs")
    args = parser.parse_args()
    collector = Collector(args.log_dirs)
    collector.collect()
