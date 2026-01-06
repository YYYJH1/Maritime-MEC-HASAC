import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import os
from harl.common.base_logger import BaseLogger


class MECLogger(BaseLogger):

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }

        datas = np.zeros(10)
        for i in range(len(self.eval_infos)):
            datas[0] += self.eval_infos[i][0].get('total_tasks', 0)
            datas[1] += self.eval_infos[i][0].get('completed_tasks', 0)
            datas[2] += self.eval_infos[i][0].get('total_computation', 0)
            datas[3] += self.eval_infos[i][0].get('uav_computation', 0)
            datas[4] += self.eval_infos[i][0].get('vessel_computation', 0)
            datas[5] += self.eval_infos[i][0].get('avg_completion_time', 0)
            datas[6] += self.eval_infos[i][0].get('avg_delay', 0)
            datas[7] += self.eval_infos[i][0].get('edge_computing_percentage', 0)
            datas[8] += 1.0 if self.eval_infos[i][0].get('queue_stability', False) else 0.0
            datas[9] += self.eval_infos[i][0].get('lyapunov_drift', 0)
            
        datas = np.array(datas)
        if len(self.eval_infos) > 0:
            datas /= len(self.eval_infos)

        completion_rate = datas[1] / max(1, datas[0])

        self.log_env(eval_env_infos)
        train_avg_rew = np.mean(self.train_episode_rewards)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        print(f"Avg completion time: {datas[5]:.2f} slots, Completion rate: {completion_rate*100:.1f}%, Queue stable: {datas[8]*100:.0f}%")
        
        self.log_file.write(
            ",".join(map(str, [
                self.total_num_steps, 
                train_avg_rew, 
                eval_avg_rew,
                datas[0],
                datas[1],
                completion_rate,
                datas[2],
                datas[3],
                datas[4],
                datas[5],
                datas[6],
                datas[7],
                datas[8],
                datas[9]
            ])) + "\n"
        )
        self.log_file.flush()
        
        if not hasattr(self, '_last_plot_step') or self.total_num_steps >= self._last_plot_step + 100000:
            self._last_plot_step = self.total_num_steps
            self._create_plots()
    
    def _create_plots(self):
        log_dir = os.path.dirname(self.log_file.name)
        plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        with open(self.log_file.name, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:
            return
            
        data = []
        for line in lines:
            try:
                values = [float(x) for x in line.strip().split(',')]
                if len(values) >= 10:
                    data.append(values)
            except:
                continue
                
        if not data:
            return
            
        data = np.array(data)
        steps = data[:, 0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data[:, 9], 'r-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Average Completion Time (slots)')
        plt.title('Average Task Completion Time')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'completion_time.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data[:, 5], 'b-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Task Completion Rate')
        plt.title('Task Completion Rate')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'completion_rate.png'))
        plt.close()
            
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data[:, 2], 'g-', label='Eval Reward', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.title('Evaluation Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'rewards.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        uav_comp = data[-1, 7]
        vessel_comp = data[-1, 8]
        plt.bar(['UAVs', 'Vessel'], [uav_comp, vessel_comp], color=['green', 'red'])
        plt.ylabel('Computation Processed')
        plt.title('Resource Utilization')
        plt.grid(axis='y')
        plt.savefig(os.path.join(plots_dir, 'resource_utilization.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data[:, 12], 'm-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Queue Stability Rate')
        plt.title('Queue Stability')
        plt.ylim([0, 1.1])
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'queue_stability.png'))
        plt.close()

    def get_task_name(self):
        return "maritime_mec"
