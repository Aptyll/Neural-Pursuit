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

// Projectile class for AI abilities
class Projectile {
    constructor(x, y, targetX, targetY, speed = 8) {
        this.x = x;
        this.y = y;
        this.radius = 8;
        this.speed = speed;
        this.life = 1.0;
        this.maxLife = 3.0; // 3 seconds
        
        // Calculate direction to target
        const dx = targetX - x;
        const dy = targetY - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        this.vx = (dx / dist) * speed;
        this.vy = (dy / dist) * speed;
        
        this.trail = [];
        this.color = '#ff6b6b';
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 1/60; // Assuming 60 FPS
        
        // Update trail
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > 10) {
            this.trail.shift();
        }
    }
    
    isExpired() {
        return this.life <= 0;
    }
    
    checkCollision(player) {
        const dx = this.x - player.x;
        const dy = this.y - player.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        return dist < this.radius + player.radius;
    }
    
    draw(ctx) {
        // Draw trail
        if (this.trail.length > 1) {
            ctx.strokeStyle = this.color + '40';
            ctx.lineWidth = this.radius;
            ctx.lineCap = 'round';
            ctx.beginPath();
            
            for (let i = 1; i < this.trail.length; i++) {
                const alpha = i / this.trail.length;
                ctx.globalAlpha = alpha * 0.5;
                
                if (i === 1) {
                    ctx.moveTo(this.trail[i-1].x, this.trail[i-1].y);
                }
                ctx.lineTo(this.trail[i].x, this.trail[i].y);
            }
            
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        
        // Draw glow
        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 2);
        gradient.addColorStop(0, this.color + '80');
        gradient.addColorStop(1, this.color + '00');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw projectile
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner glow
        const innerGradient = ctx.createRadialGradient(
            this.x - this.radius/3, 
            this.y - this.radius/3, 
            0, 
            this.x, 
            this.y, 
            this.radius
        );
        innerGradient.addColorStop(0, '#ffffff60');
        innerGradient.addColorStop(1, this.color);
        
        ctx.fillStyle = innerGradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 0.8, 0, Math.PI * 2);
        ctx.fill();
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
            speed: 3,
            isDashing: false,
            dashTime: 0,
            dashDirection: { x: 0, y: 0 }
        };
        
        // AI Abilities
        this.abilities = {
            projectile: {
                cooldown: 3000, // 3 seconds
                lastUsed: 0,
                range: 300
            },
            dash: {
                cooldown: 5000, // 5 seconds
                lastUsed: 0,
                duration: 500, // 0.5 seconds
                speed: 12
            }
        };
        
        // Projectiles array
        this.projectiles = [];
        
        // Neural network for AI (8 inputs: player pos, velocity, AI pos, cooldown states)
        this.nn = new NeuralNetwork(8, 16, 4); // 4 outputs: move_x, move_y, use_projectile, use_dash
        
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
        this.ai.isDashing = false;
        this.ai.dashTime = 0;
        
        // Reset abilities
        this.abilities.projectile.lastUsed = 0;
        this.abilities.dash.lastUsed = 0;
        this.projectiles = [];
        
        // Hide instructions and restore game cursor
        document.getElementById('instructions').classList.add('hidden');
        this.canvas.classList.remove('game-over');
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
        const currentTime = Date.now();
        
        // Check ability cooldowns
        const projectileReady = (currentTime - this.abilities.projectile.lastUsed) >= this.abilities.projectile.cooldown;
        const dashReady = (currentTime - this.abilities.dash.lastUsed) >= this.abilities.dash.cooldown;
        
        // Prepare inputs for neural network (normalized)
        const inputs = [
            this.player.x / this.canvas.width,
            this.player.y / this.canvas.height,
            this.player.vx / 10,
            this.player.vy / 10,
            this.ai.x / this.canvas.width,
            this.ai.y / this.canvas.height,
            projectileReady ? 1 : 0,
            dashReady ? 1 : 0
        ];
        
        // Get AI prediction
        const outputs = this.nn.predict(inputs);
        
        // Convert outputs to actions
        const moveX = (outputs[0] - 0.5) * 2; // -1 to 1
        const moveY = (outputs[1] - 0.5) * 2; // -1 to 1
        const useProjectile = outputs[2] > 0.7; // threshold for using projectile
        const useDash = outputs[3] > 0.8; // threshold for using dash
        
        // Handle dash ability
        if (this.ai.isDashing) {
            this.ai.dashTime -= 16; // assuming 60fps
            if (this.ai.dashTime <= 0) {
                this.ai.isDashing = false;
            } else {
                this.ai.vx = this.ai.dashDirection.x * this.abilities.dash.speed;
                this.ai.vy = this.ai.dashDirection.y * this.abilities.dash.speed;
                
                // Create dash particles
                for (let i = 0; i < 3; i++) {
                    this.particles.push(new Particle(
                        this.ai.x + (Math.random() - 0.5) * 20,
                        this.ai.y + (Math.random() - 0.5) * 20,
                        '#ffaa00',
                        {
                            x: (Math.random() - 0.5) * 8,
                            y: (Math.random() - 0.5) * 8
                        }
                    ));
                }
            }
        } else {
            // Normal movement
            const currentSpeed = this.ai.speed;
            this.ai.vx = moveX * currentSpeed;
            this.ai.vy = moveY * currentSpeed;
            
            // Use abilities based on neural network decision
            if (useProjectile && projectileReady) {
                this.useProjectile();
            }
            
            if (useDash && dashReady && !this.ai.isDashing) {
                this.useDash(moveX, moveY);
            }
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
        const futurePlayerX = this.player.x + this.player.vx * 15;
        const futurePlayerY = this.player.y + this.player.vy * 15;
        const distToPlayer = Math.sqrt(
            (this.player.x - this.ai.x) ** 2 + (this.player.y - this.ai.y) ** 2
        );
        
        // Calculate optimal actions for training
        const optimalMoveX = Math.min(1, Math.max(-1, (futurePlayerX - this.ai.x) / 100));
        const optimalMoveY = Math.min(1, Math.max(-1, (futurePlayerY - this.ai.y) / 100));
        const shouldUseProjectile = distToPlayer < this.abilities.projectile.range && distToPlayer > 50 ? 0.9 : 0.1;
        const shouldUseDash = distToPlayer > 150 && distToPlayer < 250 ? 0.9 : 0.1;
        
        this.trainingData.push({
            inputs: inputs,
            targets: [
                (optimalMoveX + 1) / 2, // normalize to 0-1
                (optimalMoveY + 1) / 2, // normalize to 0-1
                shouldUseProjectile,
                shouldUseDash
            ]
        });
        
        if (this.trainingData.length > this.maxTrainingData) {
            this.trainingData.shift();
        }
        
        // Train the network periodically
        if (this.trainingData.length > 15 && Math.random() < 0.08) {
            let totalError = 0;
            const sampleSize = Math.min(8, this.trainingData.length);
            
            for (let i = 0; i < sampleSize; i++) {
                const data = this.trainingData[this.trainingData.length - 1 - i];
                const error = this.nn.train(data.inputs, data.targets);
                totalError += error;
            }
            
            this.aiAccuracy = Math.max(0, Math.min(100, 100 - (totalError / sampleSize) * 150));
        }
    }
    
    useProjectile() {
        const currentTime = Date.now();
        this.abilities.projectile.lastUsed = currentTime;
        
        // Predict where player will be
        const futureX = this.player.x + this.player.vx * 20;
        const futureY = this.player.y + this.player.vy * 20;
        
        this.projectiles.push(new Projectile(this.ai.x, this.ai.y, futureX, futureY));
        
        // Create muzzle flash effect
        for (let i = 0; i < 8; i++) {
            this.particles.push(new Particle(
                this.ai.x,
                this.ai.y,
                '#ff6b6b',
                {
                    x: (Math.random() - 0.5) * 10,
                    y: (Math.random() - 0.5) * 10
                }
            ));
        }
    }
    
    useDash(dirX, dirY) {
        const currentTime = Date.now();
        this.abilities.dash.lastUsed = currentTime;
        this.ai.isDashing = true;
        this.ai.dashTime = this.abilities.dash.duration;
        
        // Normalize direction
        const length = Math.sqrt(dirX * dirX + dirY * dirY);
        if (length > 0) {
            this.ai.dashDirection.x = dirX / length;
            this.ai.dashDirection.y = dirY / length;
        } else {
            // Default direction towards player
            const dx = this.player.x - this.ai.x;
            const dy = this.player.y - this.ai.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            this.ai.dashDirection.x = dx / dist;
            this.ai.dashDirection.y = dy / dist;
        }
    }
    
    updateProjectiles() {
        for (let i = this.projectiles.length - 1; i >= 0; i--) {
            const projectile = this.projectiles[i];
            projectile.update();
            
            // Check collision with player
            if (projectile.checkCollision(this.player)) {
                this.gameOver('Hit by projectile!');
                return;
            }
            
            // Check if projectile is out of bounds or expired
            if (projectile.isExpired() || 
                projectile.x < 0 || projectile.x > this.canvas.width ||
                projectile.y < 0 || projectile.y > this.canvas.height) {
                this.projectiles.splice(i, 1);
            }
        }
    }
    
    checkCollision() {
        const dx = this.player.x - this.ai.x;
        const dy = this.player.y - this.ai.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < this.player.radius + this.ai.radius) {
            this.gameOver('Caught by AI!');
        }
    }
    
    gameOver(reason) {
        this.isRunning = false;
        
        // Show cursor when game ends
        this.canvas.classList.add('game-over');
        
        document.getElementById('instructions').classList.remove('hidden');
        document.getElementById('instructions').innerHTML = `
            <h2>Game Over!</h2>
            <p>${reason}</p>
            <p>Final Score: ${Math.floor(this.score)}</p>
            <p>AI Learning Progress: ${Math.round(this.aiAccuracy)}%</p>
            <p>The AI learned your patterns!</p>
            <button id="start-btn" onclick="game.start()">Play Again</button>
        `;
    }
    
    updateAbilityUI() {
        const currentTime = Date.now();
        
        // Update projectile ability UI
        const projectileSlot = document.getElementById('projectile-ability');
        const projectileCooldownLeft = Math.max(0, this.abilities.projectile.cooldown - (currentTime - this.abilities.projectile.lastUsed));
        const projectileProgress = (projectileCooldownLeft / this.abilities.projectile.cooldown) * 100;
        
        if (projectileCooldownLeft > 0) {
            projectileSlot.classList.add('cooldown');
            projectileSlot.classList.remove('active');
            const overlay = projectileSlot.querySelector('.cooldown-overlay');
            overlay.style.setProperty('--progress', `${projectileProgress}%`);
            const timer = projectileSlot.querySelector('.cooldown-timer');
            timer.textContent = Math.ceil(projectileCooldownLeft / 1000);
        } else {
            projectileSlot.classList.remove('cooldown');
            projectileSlot.classList.add('active');
            const timer = projectileSlot.querySelector('.cooldown-timer');
            timer.textContent = '';
        }
        
        // Update dash ability UI
        const dashSlot = document.getElementById('dash-ability');
        const dashCooldownLeft = Math.max(0, this.abilities.dash.cooldown - (currentTime - this.abilities.dash.lastUsed));
        const dashProgress = (dashCooldownLeft / this.abilities.dash.cooldown) * 100;
        
        if (dashCooldownLeft > 0) {
            dashSlot.classList.add('cooldown');
            dashSlot.classList.remove('active');
            const overlay = dashSlot.querySelector('.cooldown-overlay');
            overlay.style.setProperty('--progress', `${dashProgress}%`);
            const timer = dashSlot.querySelector('.cooldown-timer');
            timer.textContent = Math.ceil(dashCooldownLeft / 1000);
        } else {
            dashSlot.classList.remove('cooldown');
            dashSlot.classList.add('active');
            const timer = dashSlot.querySelector('.cooldown-timer');
            timer.textContent = '';
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
        this.updateProjectiles();
        this.checkCollision();
        this.updateParticles();
        this.updateAbilityUI();
        
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
        
        if (!this.isRunning) {
            // Show static network when not running
            ctx.fillStyle = '#333';
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Neural Network', width/2, height/2 - 10);
            ctx.fillText('(Inactive)', width/2, height/2 + 10);
            return;
        }
        
        // Get current neural network state
        const currentTime = Date.now();
        const projectileReady = (currentTime - this.abilities.projectile.lastUsed) >= this.abilities.projectile.cooldown;
        const dashReady = (currentTime - this.abilities.dash.lastUsed) >= this.abilities.dash.cooldown;
        
        const inputs = [
            this.player.x / this.canvas.width,
            this.player.y / this.canvas.height,
            this.player.vx / 10,
            this.player.vy / 10,
            this.ai.x / this.canvas.width,
            this.ai.y / this.canvas.height,
            projectileReady ? 1 : 0,
            dashReady ? 1 : 0
        ];
        
        const outputs = this.nn.predict(inputs);
        
        // Draw network visualization with real-time data
        const layers = [this.nn.inputSize, this.nn.hiddenSize, this.nn.outputSize];
        const layerWidth = width / (layers.length + 1);
        
        const inputLabels = ['Px', 'Py', 'Vx', 'Vy', 'Ax', 'Ay', 'P?', 'D?'];
        const outputLabels = ['Mx', 'My', 'Shoot', 'Dash'];
        
        for (let l = 0; l < layers.length; l++) {
            const x = layerWidth * (l + 1);
            const nodeHeight = height / (layers[l] + 1);
            
            for (let n = 0; n < layers[l]; n++) {
                const y = nodeHeight * (n + 1);
                
                // Calculate node activation intensity
                let activation = 0.5; // default for hidden layer
                if (l === 0) {
                    activation = Math.abs(inputs[n]); // input layer
                } else if (l === layers.length - 1) {
                    activation = outputs[n]; // output layer
                }
                
                // Draw connections to next layer with varying intensity
                if (l < layers.length - 1) {
                    const nextLayerX = layerWidth * (l + 2);
                    const nextNodeHeight = height / (layers[l + 1] + 1);
                    
                    for (let nn = 0; nn < layers[l + 1]; nn++) {
                        const nextY = nextNodeHeight * (nn + 1);
                        
                        // Connection strength based on weights (simplified visualization)
                        const weight = l === 0 ? this.nn.weightsIH[nn][n] : this.nn.weightsHO[nn][n];
                        const intensity = Math.min(1, Math.abs(weight) * activation);
                        const alpha = Math.floor(intensity * 255).toString(16).padStart(2, '0');
                        
                        ctx.strokeStyle = weight > 0 ? `#4a9eff${alpha}` : `#ff4a4a${alpha}`;
                        ctx.lineWidth = 1 + intensity;
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(nextLayerX, nextY);
                        ctx.stroke();
                    }
                }
                
                // Draw node with activation-based intensity
                const baseColor = l === 0 ? '#4a9eff' : l === layers.length - 1 ? '#ff4a4a' : '#888';
                const intensity = Math.min(1, Math.max(0.2, activation));
                const alpha = Math.floor(intensity * 255).toString(16).padStart(2, '0');
                
                // Node glow effect based on activation
                if (intensity > 0.5) {
                    ctx.fillStyle = baseColor + '40';
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, Math.PI * 2);
                    ctx.fill();
                }
                
                ctx.fillStyle = baseColor + alpha;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw labels for input and output nodes
                ctx.fillStyle = '#fff';
                ctx.font = '8px monospace';
                ctx.textAlign = 'center';
                
                if (l === 0 && n < inputLabels.length) {
                    ctx.fillText(inputLabels[n], x, y - 12);
                } else if (l === layers.length - 1 && n < outputLabels.length) {
                    ctx.fillText(outputLabels[n], x, y + 18);
                    
                    // Show output values
                    const value = (outputs[n] * 100).toFixed(0);
                    ctx.fillStyle = outputs[n] > 0.7 ? '#ffff00' : '#666';
                    ctx.fillText(`${value}%`, x, y + 28);
                }
            }
        }
        
        // Draw real-time decision indicators
        ctx.fillStyle = '#fff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        
        const decisions = [];
        if (outputs[2] > 0.7) decisions.push('🎯 SHOOT');
        if (outputs[3] > 0.8) decisions.push('⚡ DASH');
        if (this.ai.isDashing) decisions.push('💨 DASHING');
        
        if (decisions.length > 0) {
            ctx.fillStyle = '#ffff00';
            ctx.fillText(decisions.join(' '), 5, height - 5);
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
            // Draw projectiles
            this.projectiles.forEach(projectile => projectile.draw(this.ctx));
            
            // Draw AI prediction line
            const currentTime = Date.now();
            const projectileReady = (currentTime - this.abilities.projectile.lastUsed) >= this.abilities.projectile.cooldown;
            const dashReady = (currentTime - this.abilities.dash.lastUsed) >= this.abilities.dash.cooldown;
            
            const inputs = [
                this.player.x / this.canvas.width,
                this.player.y / this.canvas.height,
                this.player.vx / 10,
                this.player.vy / 10,
                this.ai.x / this.canvas.width,
                this.ai.y / this.canvas.height,
                projectileReady ? 1 : 0,
                dashReady ? 1 : 0
            ];
            
            const outputs = this.nn.predict(inputs);
            const moveX = (outputs[0] - 0.5) * 2;
            const moveY = (outputs[1] - 0.5) * 2;
            const predictX = this.ai.x + moveX * 50;
            const predictY = this.ai.y + moveY * 50;
            
            this.ctx.strokeStyle = '#ff4a4a20';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(this.ai.x, this.ai.y);
            this.ctx.lineTo(predictX, predictY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Draw dash effect
            if (this.ai.isDashing) {
                this.ctx.strokeStyle = '#ffaa0080';
                this.ctx.lineWidth = this.ai.radius * 2;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.x - this.ai.vx, this.ai.y - this.ai.vy);
                this.ctx.lineTo(this.ai.x, this.ai.y);
                this.ctx.stroke();
            }
            
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
