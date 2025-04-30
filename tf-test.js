import { useState } from "react";
import * as tf from "@tensorflow/tfjs";

interface IrisData {
  features: number[];
  label: IrisSpecies;
}

type IrisSpecies = "setosa" | "versicolor" | "virginica";

// 在文件顶部添加类型定义和常量
const SPECIES: IrisSpecies[] = ["setosa", "versicolor", "virginica"];
const NUM_CLASSES = SPECIES.length;

// 提取模型配置常量
const MODEL_CONFIG = {
  hiddenUnits: 6,
  learningRate: 0.05,
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
};

// 提取数据解析函数
const parseIrisData = (data: string): IrisData[] => {
  return data
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => {
      const [sepalL, sepalW, petalL, petalW, species] = line.split(",");
      return {
        features: [sepalL, sepalW, petalL, petalW].map(Number),
        label: species.trim() as IrisSpecies,
      };
    });
};

// 提取Tensor转换函数
const convertToTensor = (data: IrisData[]) => ({
  xs: tf.tensor2d(data.map((d) => d.features)),
  ys: tf.oneHot(
    tf.tensor1d(
      data.map((d) => SPECIES.indexOf(d.label)),
      "int32"
    ),
    NUM_CLASSES
  ),
});

function App() {
  const [model, setModel] = useState<tf.Sequential>();
  const [prediction, setPrediction] = useState<{
    label: string;
    confidence: number;
  }>();
  const [trainingLogs, setTrainingLogs] = useState<
    { epoch: number; loss: number; accuracy: number }[]
  >([]);
  const [testSamples, setTestSamples] = useState<IrisData[]>([]);
  const [selectedTestIndex, setSelectedTestIndex] = useState<number>(-1);
  const [loadedModel, setLoadedModel] = useState<tf.LayersModel>();
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [loadError, setLoadError] = useState<string>();
  const [selectedModelType, setSelectedModelType] = useState<
    "trained" | "loaded"
  >("trained");

  // 添加样式常量
  const styles = {
    container: {
      maxWidth: 1200,
      margin: "0 auto",
      padding: 40,
      fontFamily: "'Segoe UI', sans-serif",
    },
    title: {
      color: "#2c3e50",
      textAlign: "center",
      marginBottom: 40,
    },
    controlSection: {
      display: "flex",
      gap: 16,
      alignItems: "center",
      marginBottom: 40,
    },
    trainButton: {
      backgroundColor: "#3498db",
      color: "white",
      border: "none",
      padding: "12px 24px",
      borderRadius: 4,
      cursor: "pointer",
    },
    statusIndicator: {
      width: 10,
      height: 10,
      backgroundColor: "#27ae60",
      borderRadius: "50%",
    },
    section: {
      backgroundColor: "#f8f9fa",
      padding: 24,
      borderRadius: 8,
      marginBottom: 40,
      boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    },
    logItem: {
      display: "flex",
      gap: 16,
    },
    sampleSelect: {
      padding: 8,
      border: "1px solid #ddd",
      borderRadius: 4,
      minWidth: 250,
      marginTop: 16,
    },
    predictionResult: (isCorrect: boolean) => ({
      marginTop: 16,
      padding: 16,
      borderRadius: 4,
      backgroundColor: "#fff",
      borderLeft: `4px solid ${isCorrect ? "#27ae60" : "#e74c3c"}`,
    }),
    comparison: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 16,
      marginTop: 16,
    },
    valueText: {
      fontWeight: "bold",
      color: "#2c3e50",
      fontSize: "1.1rem",
    },
    loadButton: {
      backgroundColor: "#9b59b6",
      color: "white",
      border: "none",
      padding: "12px 24px",
      borderRadius: 4,
      cursor: "pointer",
    },
    errorText: {
      color: "#e74c3c",
      marginLeft: 8,
    },
  };

  // 新增公共数据加载函数
  const loadAndPrepareData = async () => {
    const response = await fetch("src/assets/iris.txt");
    const data = await response.text();
    const parsedData = parseIrisData(data);

    tf.util.shuffle(parsedData);
    if (testSamples.length === 0) {
      setTestSamples(parsedData.slice(140));
      setSelectedTestIndex(0);
    }

    return parsedData;
  };

  // 修改后的loadData函数
  const loadData = async () => {
    const parsedData = await loadAndPrepareData();
    return convertToTensor(parsedData.slice(0, 140));
  };

  // 优化后的模型创建函数
  const createModel = () => {
    return tf.sequential({
      layers: [
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
          inputShape: [4],
        }),
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
        }),
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
        }),
        tf.layers.dense({
          units: NUM_CLASSES,
          activation: "softmax",
        }),
      ],
    });
  };

  const trainModel = async () => {
    setTrainingLogs([]);
    const model = createModel();

    model.compile({
      optimizer: tf.train.adam(MODEL_CONFIG.learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    const train = await loadData();

    await model.fit(train.xs, train.ys, {
      epochs: MODEL_CONFIG.epochs,
      batchSize: MODEL_CONFIG.batchSize,
      validationSplit: MODEL_CONFIG.validationSplit,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setTrainingLogs((prev) => [
            ...prev,
            {
              epoch: epoch + 1,
              loss: Number(logs.loss?.toFixed(4)) || 0,
              accuracy: Number(logs.acc?.toFixed(4)) || 0, // 修正为正确的指标名称
            },
          ]);
        },
      },
    });

    setModel(model);
  };

  // 修改后的loadExternalModel函数
  const loadExternalModel = async () => {
    setIsLoadingModel(true);
    setLoadError(undefined);

    try {
      const model = await tf.loadLayersModel("src/assets/iris-model.json");
      const parsedData = await loadAndPrepareData();

      if (parsedData.length > 0) {
        predict(parsedData[140].features);
      }

      setLoadedModel(model);
      alert("模型加载成功！");
    } catch (error) {
      console.error("模型加载失败:", error);
      setLoadError("加载失败，请检查模型文件是否存在");
    }
    setIsLoadingModel(false);
  };

  const predict = (features: number[], modelType = selectedModelType) => {
    const currentModel = modelType === "trained" ? model : loadedModel;
    if (!currentModel) return;

    const input = tf.tensor2d([features]);
    const prediction = currentModel.predict(input) as tf.Tensor; // 使用的是分类模型，所以得到了各个分类的概率
    const probabilities = Array.from(prediction.dataSync()); // 将概率转化为普通数组
    const predictedIndex = prediction.argMax(1).dataSync()[0]; // 获取概率最大的分类的索引

    setPrediction({
      label: SPECIES[predictedIndex],
      confidence: Math.round(probabilities[predictedIndex] * 100),
    });

    tf.dispose(input);
  };

  // 添加模型导出方法
  const exportModel = async () => {
    if (!model) return;

    // 保存模型为文件下载
    await model.save("downloads://iris-model");
  };

  // 提取状态指示器组件
  const StatusIndicator = ({
    color,
    label,
  }: {
    color: string;
    label: string;
  }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ ...styles.statusIndicator, backgroundColor: color }} />
      <span>{label}</span>
    </div>
  );

  return (
    <div style={styles.container}>
      <h1>鸢尾花分类器</h1>
      <div style={styles.controlSection}>
        <button style={styles.trainButton} onClick={trainModel}>
          {model ? "重新训练模型" : "开始训练"}
        </button>

        {model && (
          <>
            <button
              style={{ ...styles.trainButton, backgroundColor: "#2ecc71" }}
              onClick={exportModel}
            >
              导出模型
            </button>
            <StatusIndicator color="#27ae60" label="模型已就绪" />
          </>
        )}

        <button
          style={styles.loadButton}
          onClick={loadExternalModel}
          disabled={isLoadingModel}
        >
          {isLoadingModel ? "加载模型中..." : "加载外部模型"}
        </button>

        {loadError && <span style={styles.errorText}>{loadError}</span>}
        {loadedModel && (
          <StatusIndicator color="#9b59b6" label="外部模型已加载" />
        )}
      </div>

      <div style={styles.section}>
        <h2>训练进度</h2>
        <div style={{ maxHeight: "200px", overflowY: "auto" }}>
          {trainingLogs.map((log) => (
            <div key={log.epoch} style={styles.logItem}>
              <div>轮数 {log.epoch}</div>
              <div>损失: {log.loss}</div>
              <div>准确率: {(log.accuracy * 100).toFixed(1)}%</div>
            </div>
          ))}
          {!trainingLogs.length && <div>训练尚未开始</div>}
        </div>
      </div>

      <div style={styles.section}>
        <h2>模型测试</h2>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <div>
            <label>选择测试模型：</label>
            <select
              style={styles.sampleSelect}
              value={selectedModelType}
              onChange={(e) => {
                const type = e.target.value as "trained" | "loaded";
                setSelectedModelType(type);
                if (testSamples.length > 0 && selectedTestIndex !== -1) {
                  predict(testSamples[selectedTestIndex].features, type);
                }
              }}
            >
              <option value="trained" disabled={!model}>
                自训练模型{model ? "" : "（未训练）"}
              </option>
              <option value="loaded" disabled={!loadedModel}>
                外部模型{loadedModel ? "" : "（未加载）"}
              </option>
            </select>
          </div>
          <div>
            <label>选择测试样本：</label>
            <select
              style={styles.sampleSelect}
              value={selectedTestIndex}
              onChange={(e) => {
                const index = Number(e.target.value);
                setSelectedTestIndex(index);
                if (testSamples.length > 0 && index !== -1) {
                  predict(testSamples[index].features, selectedModelType);
                }
              }}
              disabled={!testSamples.length}
            >
              {testSamples.length > 0 ? (
                testSamples.map((sample, index) => (
                  <option key={index} value={index}>
                    样本 #{index + 1} ({sample.label})
                  </option>
                ))
              ) : (
                <option value="-1">请先开始训练</option>
              )}
            </select>
          </div>
        </div>

        {prediction && (
          <div
            style={styles.predictionResult(
              prediction.label === testSamples[selectedTestIndex].label
            )}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <h3>
                预测结果（使用
                {selectedModelType === "trained" ? "自训练" : "外部"}模型）
              </h3>
              <span>{prediction.confidence}% 置信度</span>
            </div>
            <div style={styles.comparison}>
              <div>
                <label>模型预测</label>
                <div style={styles.valueText}>{prediction.label}</div>
              </div>
              <div>
                <label>实际品种</label>
                <div style={styles.valueText}>
                  {testSamples[selectedTestIndex].label}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

