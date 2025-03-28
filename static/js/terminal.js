// Global variables
let commandHistory = [];
let currentHistoryIndex = -1;
let sessionId = null;

// DOM Elements
const terminalOutput = document.getElementById('terminal-output');
const commandInput = document.getElementById('command-input');
const executeBtn = document.getElementById('execute-btn');
const currentCommand = document.getElementById('current-command');
const commandHistoryDiv = document.getElementById('command-history');
const aiAnalysis = document.getElementById('ai-analysis');
const historyList = document.getElementById('history-list');
const copyAnalysisBtn = document.getElementById('copy-analysis-btn');

// Initialize dark mode based on user preference
function initDarkMode() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.classList.add('dark');
    }
    
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        if (event.matches) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    });
}

// Terminal functionality
function focusCommandInput() {
    commandInput.focus();
}

function scrollToBottom() {
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function clearTerminal() {
    commandHistoryDiv.innerHTML = '';
    updateCommandLine();
    scrollToBottom();
}

function updateCommandLine() {
    currentCommand.textContent = commandInput.value;
}

function addToHistory(command) {
    // Skip adding if it's the same as the last command
    if (commandHistory.length > 0 && commandHistory[commandHistory.length - 1].command === command) {
        return;
    }
    
    const timestamp = new Date().toLocaleTimeString();
    const historyItem = { command, timestamp };
    commandHistory.push(historyItem);
    
    // Update history UI
    updateHistoryList();
    
    // Reset history navigation index
    currentHistoryIndex = commandHistory.length;
}

function updateHistoryList() {
    if (commandHistory.length === 0) {
        historyList.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-sm">No commands executed yet.</p>';
        return;
    }
    
    historyList.innerHTML = '';
    
    // Show only the last 10 commands
    const startIndex = Math.max(0, commandHistory.length - 10);
    for (let i = startIndex; i < commandHistory.length; i++) {
        const item = commandHistory[i];
        const historyEntry = document.createElement('div');
        historyEntry.className = 'flex justify-between items-center text-sm p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded';
        
        const cmdText = document.createElement('span');
        cmdText.className = 'font-mono text-gray-800 dark:text-gray-300';
        cmdText.textContent = item.command;
        
        const timeText = document.createElement('span');
        timeText.className = 'text-gray-500 dark:text-gray-400 text-xs ml-2';
        timeText.textContent = item.timestamp;
        
        historyEntry.appendChild(cmdText);
        historyEntry.appendChild(timeText);
        
        // Add click to reuse command
        historyEntry.addEventListener('click', () => {
            commandInput.value = item.command;
            updateCommandLine();
            focusCommandInput();
        });
        
        historyList.appendChild(historyEntry);
    }
}

// Execute command using the Flask API
async function executeCommand() {
    const command = commandInput.value.trim();
    if (!command) return;
    
    // Clear the input field
    commandInput.value = '';
    updateCommandLine();
    
    // Add command to history
    addToHistory(command);
    
    // Display the command in the terminal
    const commandBlock = document.createElement('div');
    commandBlock.className = 'mb-4';
    commandBlock.innerHTML = `
        <div class="flex mb-1">
            <span class="text-green-400 mr-2">C:\\&gt;</span>
            <span class="text-gray-200">${escapeHtml(command)}</span>
        </div>
    `;
    
    // Add a loading indicator
    const outputBlock = document.createElement('div');
    outputBlock.className = 'cmd-output pl-5 mb-2 text-gray-300';
    outputBlock.innerHTML = `
        <div class="flex items-center text-gray-300">
            <svg class="loading-spinner w-4 h-4 mr-2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Executing command...</span>
        </div>
    `;
    
    commandBlock.appendChild(outputBlock);
    commandHistoryDiv.appendChild(commandBlock);
    scrollToBottom();
    
    // Handle clear command
    if (command.toLowerCase() === 'clear') {
        setTimeout(() => {
            clearTerminal();
        }, 500);
        return;
    }
    
    try {
        // Show loading in AI analysis panel
        const aiLoadingIndicator = document.createElement('div');
        aiLoadingIndicator.className = 'ai-loading flex items-center text-gray-500 dark:text-gray-400 mb-4';
        aiLoadingIndicator.innerHTML = `
            <svg class="loading-spinner w-4 h-4 mr-2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Preparing AI analysis...</span>
        `;
        aiAnalysis.appendChild(aiLoadingIndicator);
        
        // Execute the command via our Flask API
        const headers = {
            'Content-Type': 'application/json',
        };
        
        // Add session ID if we have one
        if (sessionId) {
            headers['X-Session-ID'] = sessionId;
        }
        
        const response = await fetch('/api/execute', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                command: command,
            }),
        });
        
        // Store session ID from response if provided
        if (response.headers.has('X-Session-ID')) {
            sessionId = response.headers.get('X-Session-ID');
        }
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to execute command');
        }
        
        // Update the terminal with the result
        outputBlock.innerHTML = `<span>${escapeHtml(data.output)}</span>`;
        
        // Get the AI analysis from our Flask API
        const analysisResponse = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': sessionId
            },
            body: JSON.stringify({
                command: command,
                output: data.output,
            }),
        });
        
        const analysisData = await analysisResponse.json();
        
        if (!analysisResponse.ok) {
            throw new Error(analysisData.error || 'Failed to get AI analysis');
        }
        
        // Remove the loading indicator
        const loadingElement = aiAnalysis.querySelector('.ai-loading');
        if (loadingElement) {
            aiAnalysis.removeChild(loadingElement);
        }
        
        // Add the analysis to the AI panel
        const analysisBlock = document.createElement('div');
        analysisBlock.className = 'analysis-block mb-6 p-3 bg-gray-50 dark:bg-gray-750 rounded-lg';
        
        const analysisHeader = document.createElement('div');
        analysisHeader.className = 'flex justify-between items-center mb-2';
        analysisHeader.innerHTML = `
            <h3 class="font-medium text-gray-800 dark:text-white">Analysis of: <span class="font-mono">${escapeHtml(command)}</span></h3>
            <span class="text-xs text-gray-500 dark:text-gray-400">${new Date().toLocaleTimeString()}</span>
        `;
        
        const analysisContent = document.createElement('div');
        analysisContent.className = 'text-sm text-gray-700 dark:text-gray-300 whitespace-pre-line';
        analysisContent.textContent = analysisData.analysis;
        
        analysisBlock.appendChild(analysisHeader);
        analysisBlock.appendChild(analysisContent);
        aiAnalysis.appendChild(analysisBlock);
        aiAnalysis.scrollTop = aiAnalysis.scrollHeight;
    } catch (error) {
        console.error("Error executing command:", error);
        outputBlock.innerHTML = `<span class="text-red-400">Error: ${escapeHtml(error.message || 'An unknown error occurred')}</span>`;
        
        // Remove any loading indicators in AI panel
        const loadingElement = aiAnalysis.querySelector('.ai-loading');
        if (loadingElement) {
            aiAnalysis.removeChild(loadingElement);
        }
        
        // Add error to AI panel
        const errorBlock = document.createElement('div');
        errorBlock.className = 'error-block mb-6 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg';
        errorBlock.innerHTML = `
            <h3 class="font-medium text-red-700 dark:text-red-400 mb-1">Error Executing Command</h3>
            <p class="text-sm text-red-600 dark:text-red-300">${escapeHtml(error.message || 'An unknown error occurred')}</p>
        `;
        aiAnalysis.appendChild(errorBlock);
        aiAnalysis.scrollTop = aiAnalysis.scrollHeight;
    }
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Event Listeners
function setupEventListeners() {
    window.addEventListener('load', focusCommandInput);
    
    executeBtn.addEventListener('click', executeCommand);
    
    commandInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            executeCommand();
        } else if (e.key === 'ArrowUp') {
            // Navigate command history (up)
            if (commandHistory.length > 0) {
                currentHistoryIndex = Math.max(0, currentHistoryIndex - 1);
                commandInput.value = commandHistory[currentHistoryIndex].command;
                updateCommandLine();
                // Move cursor to end of input
                setTimeout(() => {
                    commandInput.selectionStart = commandInput.selectionEnd = commandInput.value.length;
                }, 0);
            }
            e.preventDefault();
        } else if (e.key === 'ArrowDown') {
            // Navigate command history (down)
            if (currentHistoryIndex < commandHistory.length - 1) {
                currentHistoryIndex++;
                commandInput.value = commandHistory[currentHistoryIndex].command;
            } else {
                currentHistoryIndex = commandHistory.length;
                commandInput.value = '';
            }
            updateCommandLine();
            e.preventDefault();
        }
    });
    
    commandInput.addEventListener('input', updateCommandLine);
    
    terminalOutput.addEventListener('click', focusCommandInput);
    
    copyAnalysisBtn.addEventListener('click', () => {
        const lastAnalysis = aiAnalysis.querySelector('.analysis-block:last-child .text-sm');
        if (lastAnalysis) {
            navigator.clipboard.writeText(lastAnalysis.textContent)
                .then(() => {
                    const originalText = copyAnalysisBtn.textContent;
                    copyAnalysisBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        copyAnalysisBtn.textContent = originalText;
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                });
        }
    });
}

// Initialize with a welcome message in the AI panel
function initWelcomeMessage() {
    const welcomeAnalysis = document.createElement('div');
    welcomeAnalysis.className = 'welcome-analysis p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg mb-6';
    welcomeAnalysis.innerHTML = `
        <h3 class="font-medium text-blue-700 dark:text-blue-400 mb-2">Welcome to AI-Enhanced Terminal</h3>
        <p class="text-sm text-gray-700 dark:text-gray-300">
            This application allows you to execute Windows CMD commands and receive AI-powered analysis of the results.
        </p>
        <p class="text-sm text-gray-700 dark:text-gray-300 mt-2">
            Try running commands like <span class="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">dir</span>, 
            <span class="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">ipconfig</span>, or 
            <span class="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">systeminfo</span> to get started.
        </p>
    `;
    
    const initialMessage = aiAnalysis.querySelector('.initial-message');
    if (initialMessage) {
        initialMessage.remove();
    }
    
    aiAnalysis.appendChild(welcomeAnalysis);
}

// Initialize the terminal
function initTerminal() {
    initDarkMode();
    setupEventListeners();
    initWelcomeMessage();
}

// Run initialization
document.addEventListener('DOMContentLoaded', initTerminal);