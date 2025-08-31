# 🧠 Mini-LLM (C++ Mock Inference Engine)

**Mini-LLM** is a **minimal C++ project** that demonstrates the building blocks of a Large Language Model (LLM) inference engine.
It’s designed to be lightweight, educational, and built in under **1 hour**.

This project **does not run a real LLM**, but it mimics the pipeline:
👉 **Tokenizer → Embedding → Matrix Multiply → Softmax → Token Sampling**

---

## ✨ Features

* ✅ **Simple whitespace tokenizer**
* ✅ **Toy embedding layer** (convert token → vector)
* ✅ **Matrix multiplication + Softmax** (like transformer forward pass)
* ✅ **Random sampling from a fake vocabulary**
* ✅ **Clean C++11 implementation**, no external dependencies
* ✅ Ready for **future extensions** (quantization, PyBind11, CUDA, etc.)

---

## 📂 Project Structure

```
mini-llm/
 ├── src/
 │    └── main.cpp       # Core logic
 ├── README.md           # Documentation
 └── Makefile (optional) # Build automation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/SiddarthaKarri/mini-llm.git
cd mini-llm
```

### 2️⃣ Compile

```bash
g++ src/main.cpp -o mini-llm -std=c++11
```

### 3️⃣ Run

```bash
./mini-llm "Hello world from Sid"
```

---

## 🖥️ Example Output

```
Tokens:
0: Hello
1: world
2: from
3: Karri

Generated output: [MockResponse] Sid -> model
```

---

## 🔍 How It Works

1. **Tokenizer**
   Splits input text into tokens based on whitespace.

2. **Toy Embedding**
   Encodes the last token into a small vector (fake embedding).

3. **Linear Layer (MatMul)**
   Multiplies embedding with a random weight matrix to produce logits.

4. **Softmax**
   Converts logits into probability distribution over fake vocabulary.

5. **Sampling**
   Randomly selects the next token from vocabulary based on probabilities.

---

## 🛠️ Future Improvements

* 🔹 Replace whitespace tokenizer with a **BPE tokenizer**
* 🔹 Implement **real transformer layers** (attention, feedforward)
* 🔹 Add **quantization** for efficiency
* 🔹 Bind with **Python (PyBind11)**
* 🔹 GPU acceleration via **CUDA**

---

## 📚 Inspiration

* [llama.cpp](https://github.com/ggerganov/llama.cpp) – lightweight LLM inference in pure C++
* [YALM](https://github.com/andrewkchan/yalm) – Yet Another Language Model, educational C++/CUDA LLM

---

## 📜 License

MIT License – free to use, modify, and learn from.

---