"""
Data Analysis Utility Module
Provides tools for analyzing concentration tracking data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os


class ConcentrationDataAnalyzer:
    """Analyzes concentration tracking data from log files"""
    
    def __init__(self, log_file_path: str = "data/concentration_log.txt"):
        """
        Initialize data analyzer
        
        Args:
            log_file_path: Path to the log file
        """
        self.log_file_path = log_file_path
        self.data = None
    
    def parse_log_file(self) -> pd.DataFrame:
        """
        Parse log file and extract concentration data
        
        Returns:
            DataFrame with parsed concentration data
        """
        if not os.path.exists(self.log_file_path):
            print(f"Log file not found: {self.log_file_path}")
            return pd.DataFrame()
        
        data_rows = []
        
        try:
            with open(self.log_file_path, 'r') as file:
                for line in file:
                    if "Concentration:" in line:
                        # Parse concentration data from log line
                        row = self._parse_concentration_line(line)
                        if row:
                            data_rows.append(row)
            
            self.data = pd.DataFrame(data_rows)
            if not self.data.empty:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values('timestamp')
            
            return self.data
            
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return pd.DataFrame()
    
    def _parse_concentration_line(self, line: str) -> Optional[Dict]:
        """Parse a single concentration log line"""
        try:
            # Extract timestamp
            timestamp_str = line.split(' - ')[0]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            
            # Extract concentration score
            if "Concentration:" in line:
                parts = line.split("Concentration: ")[1]
                score_str = parts.split(" ")[0]
                score = float(score_str)
                
                # Extract level
                level_start = line.find("(") + 1
                level_end = line.find(")")
                level = line[level_start:level_end] if level_start > 0 and level_end > 0 else "Unknown"
                
                # Extract blink rate if present
                blink_rate = 0.0
                if "Blink Rate:" in line:
                    blink_parts = line.split("Blink Rate: ")[1]
                    blink_rate_str = blink_parts.split("/min")[0]
                    blink_rate = float(blink_rate_str)
                
                # Extract face detection status
                face_detected = "Face: Yes" in line
                
                return {
                    'timestamp': timestamp,
                    'concentration_score': score,
                    'concentration_level': level,
                    'blink_rate': blink_rate,
                    'face_detected': face_detected
                }
        
        except Exception as e:
            print(f"Error parsing line: {e}")
            return None
    
    def generate_summary_report(self) -> Dict:
        """
        Generate summary statistics report
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.data is None or self.data.empty:
            return {}
        
        summary = {
            'total_sessions': 1,  # Simplified for single session
            'total_duration': self._calculate_session_duration(),
            'average_concentration': self.data['concentration_score'].mean(),
            'max_concentration': self.data['concentration_score'].max(),
            'min_concentration': self.data['concentration_score'].min(),
            'concentration_std': self.data['concentration_score'].std(),
            'average_blink_rate': self.data['blink_rate'].mean(),
            'face_detection_rate': self.data['face_detected'].mean() * 100,
            'level_distribution': self.data['concentration_level'].value_counts().to_dict(),
            'concentration_trend': self._calculate_trend()
        }
        
        return summary
    
    def _calculate_session_duration(self) -> str:
        """Calculate total session duration"""
        if self.data is None or len(self.data) < 2:
            return "0:00:00"
        
        duration = self.data['timestamp'].max() - self.data['timestamp'].min()
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def _calculate_trend(self) -> str:
        """Calculate concentration trend"""
        if self.data is None or len(self.data) < 10:
            return "Insufficient data"
        
        # Compare first half vs second half
        mid_point = len(self.data) // 2
        first_half_avg = self.data['concentration_score'].iloc[:mid_point].mean()
        second_half_avg = self.data['concentration_score'].iloc[mid_point:].mean()
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.05:
            return "Improving"
        elif diff < -0.05:
            return "Declining"
        else:
            return "Stable"
    
    def plot_concentration_over_time(self, save_path: str = None):
        """
        Plot concentration score over time
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.data is None or self.data.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['timestamp'], self.data['concentration_score'], 
                linewidth=2, alpha=0.7, label='Concentration Score')
        
        # Add threshold lines
        plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Good Threshold')
        plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Alert Threshold')
        
        plt.xlabel('Time')
        plt.ylabel('Concentration Score')
        plt.title('Concentration Tracking Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_concentration_distribution(self, save_path: str = None):
        """
        Plot concentration score distribution
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.data is None or self.data.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.data['concentration_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Concentration Score')
        plt.ylabel('Frequency')
        plt.title('Concentration Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(self.data['concentration_score'])
        plt.ylabel('Concentration Score')
        plt.title('Concentration Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def export_summary_report(self, output_path: str = "data/concentration_report.txt"):
        """
        Export summary report to text file
        
        Args:
            output_path: Path to save the report
        """
        summary = self.generate_summary_report()
        
        if not summary:
            print("No data available for report generation")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as file:
            file.write("Concentration Tracking Summary Report\n")
            file.write("=" * 40 + "\n\n")
            
            file.write(f"Session Duration: {summary['total_duration']}\n")
            file.write(f"Average Concentration: {summary['average_concentration']:.3f}\n")
            file.write(f"Maximum Concentration: {summary['max_concentration']:.3f}\n")
            file.write(f"Minimum Concentration: {summary['min_concentration']:.3f}\n")
            file.write(f"Standard Deviation: {summary['concentration_std']:.3f}\n")
            file.write(f"Average Blink Rate: {summary['average_blink_rate']:.1f} blinks/min\n")
            file.write(f"Face Detection Rate: {summary['face_detection_rate']:.1f}%\n")
            file.write(f"Concentration Trend: {summary['concentration_trend']}\n\n")
            
            file.write("Concentration Level Distribution:\n")
            for level, count in summary['level_distribution'].items():
                file.write(f"  {level}: {count} measurements\n")
        
        print(f"Summary report saved to: {output_path}")


def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze concentration tracking data")
    parser.add_argument("--log-file", default="data/concentration_log.txt", 
                       help="Path to log file")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate plots")
    parser.add_argument("--report", action="store_true", 
                       help="Generate summary report")
    parser.add_argument("--output-dir", default="data/", 
                       help="Output directory for plots and reports")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ConcentrationDataAnalyzer(args.log_file)
    
    # Parse data
    print("Parsing log file...")
    data = analyzer.parse_log_file()
    
    if data.empty:
        print("No data found or failed to parse log file")
        return
    
    print(f"Parsed {len(data)} data points")
    
    # Generate plots
    if args.plot:
        print("Generating plots...")
        analyzer.plot_concentration_over_time(
            save_path=os.path.join(args.output_dir, "concentration_timeline.png")
        )
        analyzer.plot_concentration_distribution(
            save_path=os.path.join(args.output_dir, "concentration_distribution.png")
        )
    
    # Generate report
    if args.report:
        print("Generating summary report...")
        analyzer.export_summary_report(
            output_path=os.path.join(args.output_dir, "concentration_report.txt")
        )
    
    # Print summary to console
    summary = analyzer.generate_summary_report()
    print("\nSummary Statistics:")
    print(f"Session Duration: {summary['total_duration']}")
    print(f"Average Concentration: {summary['average_concentration']:.3f}")
    print(f"Concentration Trend: {summary['concentration_trend']}")


if __name__ == "__main__":
    main()

