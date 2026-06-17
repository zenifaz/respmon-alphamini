"""
Alphamini Project – Respiratory Monitor
Press ESC or Q in the window to stop.
"""
import logging
from base import RespiratoryMonitor, RespiratoryAnalyzer

logging.basicConfig(
    format="%(asctime)s :: %(levelname)s :: %(message)s",
    level=logging.INFO)

if __name__ == "__main__":
    print("=" * 55)
    print("  Alphamini Respiratory Monitor")
    print("  Press ESC or Q in the window to stop")
    print("=" * 55 + "\n")

    monitor = RespiratoryMonitor(
        capture_target=0,
        visualize='pyqtgraph',
        fps_limit=10,
        save_all_data=True,
        motion_extraction_method='average'
    )

    monitor.run()

    print("\n[*] Session ended. Analyzing…\n")
    analyzer = RespiratoryAnalyzer("respiratory_session_data.csv")
    analyzer.print_report()
    print("[+] Done. Results saved to respiratory_session_data.csv")