# NOTEBOOK ANALYSE:

path = "../featout_project/trained_models/mnist"
import os
import json

losses = [[], []]
test_accs = [[], []]
runtimes = [[], []]
for f in os.listdir(path):
    if f[-4:] != "json" or f.startswith("m_feat_1"):
        continue
    print(f)
    parts = f.split("_")
    with open(os.path.join(path, f), "r") as infile:
        res = json.load(infile)
    # print(res)
    featout = int(parts[2])
    losses[featout].append(res["losses"])
    test_accs[featout].append(res["test_acc"])
    runtimes[featout].append(res["runtimes"])

# PLOT LOSSES

plt.figure(figsize=(15, 8))
for loss_non_featout in losses[0]:
    plt.plot(loss_non_featout, c="red")
for loss_featout in losses[1]:
    plt.plot(loss_featout, c="blue")
plt.title("Losses - red no featout, blue featout")
plt.show()

# PLOT TEST ACCS

plt.figure(figsize=(15, 8))
for test_featout in test_accs[1]:
    plt.plot(test_featout, c="blue")
for test_non_featout in test_accs[0]:
    plt.plot(test_non_featout, c="red")

plt.title("Test accuracy - red no featout, blue featout")
plt.show()

#### Compare jump in performance between featout and non featout runs

featout_runs = np.where(np.mean(np.array(runtimes[1]), axis=0) > 100)[0]
non_featout_runs = np.where(np.mean(np.array(runtimes[1]), axis=0) < 100)[0]

featout_test_accs = np.array(test_accs[1])
baseline_test_accs = np.array(test_accs[0])
for run_nr in featout_runs:
    avg_perf_increase = np.mean(
        featout_test_accs[:, run_nr] - featout_test_accs[:, run_nr - 1]
    )
    bl_increase = np.mean(
        baseline_test_accs[:, run_nr] - baseline_test_accs[:, run_nr - 1]
    )
    print(
        f"featout boost in epoch {run_nr}:", round(avg_perf_increase, 2),
        "baseline:", round(bl_increase, 2)
    )

print("------- non featout --------")
for run_nr in non_featout_runs[1:]:  # exclude first epoch
    avg_perf_increase = np.mean(
        featout_test_accs[:, run_nr] - featout_test_accs[:, run_nr - 1]
    )
    bl_increase = np.mean(
        baseline_test_accs[:, run_nr] - baseline_test_accs[:, run_nr - 1]
    )
    print(
        f"featout boost in epoch {run_nr}:", round(avg_perf_increase, 2),
        "baseline:", round(bl_increase, 2)
    )
