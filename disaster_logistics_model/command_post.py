import subprocess
import time

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    try:
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:")
        print(e)

def main():
    start_time = time.time()
    scripts = [
        "main_base.py",
        "main_base_L.py",
        "main_redun.py",
        "compiled_sensitivity.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nAll scripts executed.")
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()