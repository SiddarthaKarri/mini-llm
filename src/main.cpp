#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

// ----------------------------
// Simple whitespace tokenizer
// ----------------------------
std::vector<std::string> tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// ----------------------------
// Matrix multiplication (A: m x n, B: n x p) -> C: m x p
// ----------------------------
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B) {
    size_t m = A.size();
    size_t n = A[0].size();
    size_t p = B[0].size();

    std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0f));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            for (size_t k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// ----------------------------
// Softmax
// ----------------------------
std::vector<float> softmax(const std::vector<float> &logits) {
    std::vector<float> probs(logits.size());
    float maxLogit = *max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] /= sum;
    }
    return probs;
}

// ----------------------------
// Mock inference: 
// Token -> vector -> matmul -> softmax -> sample vocab word
// ----------------------------
std::string mock_infer(const std::vector<std::string> &tokens) {
    // Fake vocab
    static std::vector<std::string> vocab = {"hello", "world", "AI", "future", "C++", "model", "token"};

    size_t vocab_size = vocab.size();
    size_t embed_dim = 4; // small embedding size

    // Random generator
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Step 1: Encode last token as fake embedding
    std::vector<std::vector<float>> input(1, std::vector<float>(embed_dim));
    for (size_t i = 0; i < embed_dim; i++) {
        input[0][i] = static_cast<float>((tokens.back()[0] + i) % 7) / 7.0f; 
    }

    // Step 2: Fake weight matrix [embed_dim x vocab_size]
    std::vector<std::vector<float>> W(embed_dim, std::vector<float>(vocab_size));
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for (size_t i = 0; i < embed_dim; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            W[i][j] = dist(gen);
        }
    }

    // Step 3: Matmul -> logits
    auto logits = matmul(input, W)[0];

    // Step 4: Softmax -> probs
    auto probs = softmax(logits);

    // Step 5: Sample word based on probs
    std::discrete_distribution<> sample(probs.begin(), probs.end());
    std::string chosen = vocab[sample(gen)];

    return "[MockResponse] " + tokens.back() + " -> " + chosen;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./mini-llm \"Your prompt here\"\n";
        return 1;
    }

    std::string input = argv[1];

    // Tokenize input
    auto tokens = tokenize(input);

    std::cout << "Tokens:\n";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << i << ": " << tokens[i] << "\n";
    }

    // Mock inference
    if (!tokens.empty()) {
        std::string output = mock_infer(tokens);
        std::cout << "\nGenerated output: " << output << "\n";
    } else {
        std::cout << "No tokens found.\n";
    }

    return 0;
}
