// Neural Network Class for AI Learning
class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Initialize weights with small random values
        this.weightsIH = this.createMatrix(this.hiddenSize, this.inputSize);
        this.weightsHO = this.createMatrix(this.outputSize, this.hiddenSize);
        this.biasH = this.createMatrix(this.hiddenSize, 1);
        this.biasO = this.createMatrix(this.outputSize, 1);
        
        this.learningRate = 0.1;
        this.momentum = 0.9;
        
        // Previous weight changes for momentum
        this.prevWeightsIH = this.createMatrix(this.hiddenSize, this.inputSize, 0);
        this.prevWeightsHO = this.createMatrix(this.outputSize, this.hiddenSize, 0);
    }
    
    createMatrix(rows, cols, fillValue = null) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = fillValue !== null ? fillValue : (Math.random() * 2 - 1) * 0.5;
            }
        }
        return matrix;
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    sigmoidDerivative(x) {
        return x * (1 - x);
    }
    
    predict(inputArray) {
        // Convert input to matrix
        const inputs = this.arrayToMatrix(inputArray);
        
        // Hidden layer
        const hidden = this.matrixMultiply(this.weightsIH, inputs);
        this.matrixAdd(hidden, this.biasH);
        this.matrixMap(hidden, this.sigmoid);
        
        // Output layer
        const output = this.matrixMultiply(this.weightsHO, hidden);
        this.matrixAdd(output, this.biasO);
        this.matrixMap(output, this.sigmoid);
        
        return this.matrixToArray(output);
    }
    
    train(inputArray, targetArray) {
        // Forward propagation
        const inputs = this.arrayToMatrix(inputArray);
        
        const hidden = this.matrixMultiply(this.weightsIH, inputs);
        this.matrixAdd(hidden, this.biasH);
        this.matrixMap(hidden, this.sigmoid);
        
        const outputs = this.matrixMultiply(this.weightsHO, hidden);
        this.matrixAdd(outputs, this.biasO);
        this.matrixMap(outputs, this.sigmoid);
        
        // Convert targets to matrix
        const targets = this.arrayToMatrix(targetArray);
        
        // Calculate output errors
        const outputErrors = this.matrixSubtract(targets, outputs);
        
        // Calculate gradients
        const gradients = this.matrixMap(outputs, this.sigmoidDerivative, true);
        this.matrixMultiplyElement(gradients, outputErrors);
        this.matrixMultiplyScalar(gradients, this.learningRate);
        
        // Calculate deltas
        const hiddenT = this.matrixTranspose(hidden);
        const weightsHODeltas = this.matrixMultiply(gradients, hiddenT);
        
        // Update weights and biases with momentum
        for (let i = 0; i < this.weightsHO.length; i++) {
            for (let j = 0; j < this.weightsHO[i].length; j++) {
                const delta = weightsHODeltas[i][j] + this.momentum * this.prevWeightsHO[i][j];
                this.weightsHO[i][j] += delta;
                this.prevWeightsHO[i][j] = delta;
            }
        }
        this.matrixAdd(this.biasO, gradients);
        
        // Calculate hidden layer errors
        const weightsHOT = this.matrixTranspose(this.weightsHO);
        const hiddenErrors = this.matrixMultiply(weightsHOT, outputErrors);
        
        // Calculate hidden gradients
        const hiddenGradient = this.matrixMap(hidden, this.sigmoidDerivative, true);
        this.matrixMultiplyElement(hiddenGradient, hiddenErrors);
        this.matrixMultiplyScalar(hiddenGradient, this.learningRate);
        
        // Calculate input->hidden deltas
        const inputsT = this.matrixTranspose(inputs);
        const weightsIHDeltas = this.matrixMultiply(hiddenGradient, inputsT);
        
        // Update input->hidden weights with momentum
        for (let i = 0; i < this.weightsIH.length; i++) {
            for (let j = 0; j < this.weightsIH[i].length; j++) {
                const delta = weightsIHDeltas[i][j] + this.momentum * this.prevWeightsIH[i][j];
                this.weightsIH[i][j] += delta;
                this.prevWeightsIH[i][j] = delta;
            }
        }
        this.matrixAdd(this.biasH, hiddenGradient);
        
        // Return error for accuracy tracking
        let totalError = 0;
        for (let i = 0; i < outputErrors.length; i++) {
            totalError += Math.abs(outputErrors[i][0]);
        }
        return totalError / outputErrors.length;
    }
    
    // Matrix operations
    arrayToMatrix(arr) {
        const matrix = [];
        for (let i = 0; i < arr.length; i++) {
            matrix[i] = [arr[i]];
        }
        return matrix;
    }
    
    matrixToArray(matrix) {
        const arr = [];
        for (let i = 0; i < matrix.length; i++) {
            arr[i] = matrix[i][0];
        }
        return arr;
    }
    
    matrixMultiply(a, b) {
        const result = this.createMatrix(a.length, b[0].length, 0);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    
    matrixAdd(a, b) {
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                a[i][j] += b[i][j];
            }
        }
    }
    
    matrixSubtract(a, b) {
        const result = this.createMatrix(a.length, a[0].length);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }
    
    matrixTranspose(matrix) {
        const result = this.createMatrix(matrix[0].length, matrix.length);
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    matrixMultiplyElement(a, b) {
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                a[i][j] *= b[i][j];
            }
        }
    }
    
    matrixMultiplyScalar(matrix, scalar) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                matrix[i][j] *= scalar;
            }
        }
    }
    
    matrixMap(matrix, fn, returnNew = false) {
        const result = returnNew ? this.createMatrix(matrix.length, matrix[0].length) : matrix;
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                const val = matrix[i][j];
                result[i][j] = fn(val);
            }
        }
        return result;
    }
}

// Particle class for visual effects
class Particle {
    constructor(x, y, color, velocity) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.velocity = velocity;
        this.life = 1.0;
        this.decay = 0.02;
        this.size = Math.random() * 3 + 1;
    }
    
    update() {
        this.x += this.velocity.x;
        this.y += this.velocity.y;
        this.life -= this.decay;
        this.velocity.x *= 0.98;
        this.velocity.y *= 0.98;
    }
    
    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.life;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

// Game class
class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.neuralCanvas = document.getElementById('neural-canvas');
        this.neuralCtx = this.neuralCanvas.getContext('2d');
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Game state
        this.isRunning = false;
        this.score = 0;
        this.aiAccuracy = 0;
        
        // Player
        this.player = {
            x: this.canvas.width / 2,
            y: this.canvas.height / 2,
            targetX: this.canvas.width / 2,
            targetY: this.canvas.height / 2,
            radius: 15,
            color: '#4a9eff',
            trail: []
        };
        
        // AI opponent
        this.ai = {
            x: this.canvas.width / 4,
            y: this.canvas.height / 4,
            vx: 0,
            vy: 0,
            radius: 15,
            color: '#ff4a4a',
            trail: [],
            speed: 3
        };
        
        // Neural network for AI (6 inputs: player pos, velocity, AI pos)
        this.nn = new NeuralNetwork(6, 12, 2);
        
        // Training data buffer
        this.trainingData = [];
        this.maxTrainingData = 100;
        
        // Particles
        this.particles = [];
        
        // Mouse tracking
        this.mouseX = this.canvas.width / 2;
        this.mouseY = this.canvas.height / 2;
        
        // Event listeners
        this.setupEventListeners();
        
        // Start animation loop
        this.lastTime = 0;
        this.animate(0);
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    setupEventListeners() {
        // Mouse movement
        this.canvas.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            
            if (this.isRunning) {
                this.player.targetX = e.clientX;
                this.player.targetY = e.clientY;
            }
        });
        
        // Start button
        document.getElementById('start-btn').addEventListener('click', () => {
            this.start();
        });
    }
    
    start() {
        this.isRunning = true;
        this.score = 0;
        this.aiAccuracy = 0;
        this.trainingData = [];
        
        // Reset positions
        this.player.x = this.canvas.width / 2;
        this.player.y = this.canvas.height / 2;
        this.ai.x = this.canvas.width / 4;
        this.ai.y = this.canvas.height / 4;
        
        // Hide instructions
        document.getElementById('instructions').classList.add('hidden');
    }
    
    updatePlayer() {
        // Smooth movement towards mouse
        const dx = this.player.targetX - this.player.x;
        const dy = this.player.targetY - this.player.y;
        
        this.player.vx = dx * 0.1;
        this.player.vy = dy * 0.1;
        
        this.player.x += this.player.vx;
        this.player.y += this.player.vy;
        
        // Keep player in bounds
        this.player.x = Math.max(this.player.radius, Math.min(this.canvas.width - this.player.radius, this.player.x));
        this.player.y = Math.max(this.player.radius, Math.min(this.canvas.height - this.player.radius, this.player.y));
        
        // Update trail
        this.player.trail.push({ x: this.player.x, y: this.player.y });
        if (this.player.trail.length > 20) {
            this.player.trail.shift();
        }
        
        // Create particles
        if (Math.random() < 0.3) {
            this.particles.push(new Particle(
                this.player.x,
                this.player.y,
                this.player.color,
                {
                    x: (Math.random() - 0.5) * 2,
                    y: (Math.random() - 0.5) * 2
                }
            ));
        }
    }
    
    updateAI() {
        // Prepare inputs for neural network (normalized)
        const inputs = [
            this.player.x / this.canvas.width,
            this.player.y / this.canvas.height,
            this.player.vx / 10,
            this.player.vy / 10,
            this.ai.x / this.canvas.width,
            this.ai.y / this.canvas.height
        ];
        
        // Get AI prediction
        const outputs = this.nn.predict(inputs);
        
        // Convert outputs to movement
        const targetX = outputs[0] * this.canvas.width;
        const targetY = outputs[1] * this.canvas.height;
        
        // Move AI towards predicted position
        const dx = targetX - this.ai.x;
        const dy = targetY - this.ai.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist > 0) {
            this.ai.vx = (dx / dist) * this.ai.speed;
            this.ai.vy = (dy / dist) * this.ai.speed;
        }
        
        this.ai.x += this.ai.vx;
        this.ai.y += this.ai.vy;
        
        // Keep AI in bounds
        this.ai.x = Math.max(this.ai.radius, Math.min(this.canvas.width - this.ai.radius, this.ai.x));
        this.ai.y = Math.max(this.ai.radius, Math.min(this.canvas.height - this.ai.radius, this.ai.y));
        
        // Update trail
        this.ai.trail.push({ x: this.ai.x, y: this.ai.y });
        if (this.ai.trail.length > 20) {
            this.ai.trail.shift();
        }
        
        // Create particles
        if (Math.random() < 0.3) {
            this.particles.push(new Particle(
                this.ai.x,
                this.ai.y,
                this.ai.color,
                {
                    x: (Math.random() - 0.5) * 2,
                    y: (Math.random() - 0.5) * 2
                }
            ));
        }
        
        // Collect training data
        const futurePlayerX = this.player.x + this.player.vx * 10;
        const futurePlayerY = this.player.y + this.player.vy * 10;
        
        this.trainingData.push({
            inputs: inputs,
            targets: [
                futurePlayerX / this.canvas.width,
                futurePlayerY / this.canvas.height
            ]
        });
        
        if (this.trainingData.length > this.maxTrainingData) {
            this.trainingData.shift();
        }
        
        // Train the network periodically
        if (this.trainingData.length > 10 && Math.random() < 0.1) {
            let totalError = 0;
            const sampleSize = Math.min(10, this.trainingData.length);
            
            for (let i = 0; i < sampleSize; i++) {
                const data = this.trainingData[this.trainingData.length - 1 - i];
                const error = this.nn.train(data.inputs, data.targets);
                totalError += error;
            }
            
            this.aiAccuracy = Math.max(0, Math.min(100, 100 - (totalError / sampleSize) * 200));
        }
    }
    
    checkCollision() {
        const dx = this.player.x - this.ai.x;
        const dy = this.player.y - this.ai.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < this.player.radius + this.ai.radius) {
            // Game over
            this.isRunning = false;
            document.getElementById('instructions').classList.remove('hidden');
            document.getElementById('instructions').innerHTML = `
                <h2>Game Over!</h2>
                <p>Final Score: ${this.score}</p>
                <p>AI Learning Progress: ${Math.round(this.aiAccuracy)}%</p>
                <p>The AI learned your patterns!</p>
                <button id="start-btn" onclick="game.start()">Play Again</button>
            `;
        }
    }
    
    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            particle.update();
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }
    
    update(deltaTime) {
        if (!this.isRunning) return;
        
        this.updatePlayer();
        this.updateAI();
        this.checkCollision();
        this.updateParticles();
        
        // Update score
        this.score += deltaTime;
        
        // Update UI
        document.getElementById('player-score').textContent = Math.floor(this.score);
        document.getElementById('ai-accuracy').textContent = Math.round(this.aiAccuracy) + '%';
    }
    
    drawOrb(x, y, radius, color, trail) {
        // Draw trail
        if (trail.length > 1) {
            this.ctx.strokeStyle = color + '40';
            this.ctx.lineWidth = radius * 2;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            
            for (let i = 1; i < trail.length; i++) {
                const alpha = i / trail.length;
                this.ctx.globalAlpha = alpha * 0.3;
                
                if (i === 1) {
                    this.ctx.moveTo(trail[i-1].x, trail[i-1].y);
                }
                this.ctx.lineTo(trail[i].x, trail[i].y);
            }
            
            this.ctx.stroke();
            this.ctx.globalAlpha = 1;
        }
        
        // Draw glow
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius * 3);
        gradient.addColorStop(0, color + '40');
        gradient.addColorStop(1, color + '00');
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius * 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw orb
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Inner glow
        const innerGradient = this.ctx.createRadialGradient(x - radius/3, y - radius/3, 0, x, y, radius);
        innerGradient.addColorStop(0, '#ffffff40');
        innerGradient.addColorStop(1, color);
        
        this.ctx.fillStyle = innerGradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius * 0.9, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    drawNeuralNetwork() {
        const ctx = this.neuralCtx;
        const width = this.neuralCanvas.width;
        const height = this.neuralCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw network visualization
        const layers = [this.nn.inputSize, this.nn.hiddenSize, this.nn.outputSize];
        const layerWidth = width / (layers.length + 1);
        
        for (let l = 0; l < layers.length; l++) {
            const x = layerWidth * (l + 1);
            const nodeHeight = height / (layers[l] + 1);
            
            for (let n = 0; n < layers[l]; n++) {
                const y = nodeHeight * (n + 1);
                
                // Draw connections to next layer
                if (l < layers.length - 1) {
                    const nextLayerX = layerWidth * (l + 2);
                    const nextNodeHeight = height / (layers[l + 1] + 1);
                    
                    for (let nn = 0; nn < layers[l + 1]; nn++) {
                        const nextY = nextNodeHeight * (nn + 1);
                        
                        ctx.strokeStyle = '#ffffff10';
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(nextLayerX, nextY);
                        ctx.stroke();
                    }
                }
                
                // Draw node
                const nodeColor = l === 0 ? '#4a9eff' : l === layers.length - 1 ? '#ff4a4a' : '#666';
                ctx.fillStyle = nodeColor;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }
    
    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid pattern
        this.ctx.strokeStyle = '#111';
        this.ctx.lineWidth = 1;
        const gridSize = 50;
        
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Draw particles
        this.particles.forEach(particle => particle.draw(this.ctx));
        
        if (this.isRunning) {
            // Draw AI prediction line
            const inputs = [
                this.player.x / this.canvas.width,
                this.player.y / this.canvas.height,
                this.player.vx / 10,
                this.player.vy / 10,
                this.ai.x / this.canvas.width,
                this.ai.y / this.canvas.height
            ];
            
            const outputs = this.nn.predict(inputs);
            const predictX = outputs[0] * this.canvas.width;
            const predictY = outputs[1] * this.canvas.height;
            
            this.ctx.strokeStyle = '#ff4a4a20';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(this.ai.x, this.ai.y);
            this.ctx.lineTo(predictX, predictY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Draw orbs
            this.drawOrb(this.player.x, this.player.y, this.player.radius, this.player.color, this.player.trail);
            this.drawOrb(this.ai.x, this.ai.y, this.ai.radius, this.ai.color, this.ai.trail);
        }
        
        // Draw neural network visualization
        this.drawNeuralNetwork();
    }
    
    animate(currentTime) {
        const deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;
        
        this.update(deltaTime);
        this.draw();
        
        requestAnimationFrame((time) => this.animate(time));
    }
}

// Initialize game when page loads
let game;
window.addEventListener('load', () => {
    game = new Game();
});
