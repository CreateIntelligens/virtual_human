$(function(){
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('sessionId') || document.getElementById('sessionid').value || '0';
    document.getElementById('sessionInfo').textContent = `Session ID: ${sessionId}`;
    document.getElementById('sessionid').value = sessionId;

    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let dialogHistory = [];
    let checkSpeakingInterval;
    let currentDisplayName = '';
    let canvas, ctx;
    let isGreenScreenEnabled = false;
    let animationFrameId = null;
    let isProcessing = false;
    let sdk = null;

    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    
    const videoElement = document.getElementById('rtc_media_player');
    const chatHistory = document.getElementById('chatHistory');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const errorMessage = document.getElementById('errorMessage');
    const permissionModal = document.getElementById('permissionModal');
    const allowButton = document.getElementById('allowButton');
    const denyButton = document.getElementById('denyButton');
    const transcriptionResult = document.getElementById('transcriptionResult');

    function switchToEnglishInput() {
        if (isMac) {
            const notification = document.getElementById('inputMethodNotification');
            notification.style.display = 'block';
            
            try {
                const script = `
                    tell application "System Events"
                        key code 50 using {control down, command down}
                    end tell
                `;
                console.log('Attempting to switch input method to English');
            } catch (error) {
                console.error('Failed to switch input method:', error);
            }
        }
    }

    if (isMac) {
        switchToEnglishInput();
        document.addEventListener('compositionstart', () => {
            document.getElementById('inputMethodNotification').style.display = 'block';
        });
        document.addEventListener('compositionend', () => {
            setTimeout(() => {
                document.getElementById('inputMethodNotification').style.display = 'none';
            }, 3000);
        });
    }

    function showError(message) {
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
    }

    function initializeGreenScreen(videoElement) {
        canvas = document.getElementById('outputCanvas');
        ctx = canvas.getContext('2d');
        
        const aspectRatio = videoElement.videoHeight / videoElement.videoWidth;
        canvas.width = 600;
        canvas.height = Math.round(600 * aspectRatio);

        if (isGreenScreenEnabled) {
            $('#outputCanvas').show();
            $('#rtc_media_player').css('opacity', '0');
            processFrame();
        }
        console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' Green screen initialized');
    }

    function processFrame() {
        if (!isGreenScreenEnabled || isProcessing) {
            return;
        }

        isProcessing = true;
        
        if (videoElement.paused || videoElement.ended) {
            isProcessing = false;
            return;
        }

        try {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const newWidth = videoElement.videoWidth;
            const newHeight = videoElement.videoHeight;
            
            const x = (canvas.width - newWidth) / 2;
            const y = (canvas.height - newHeight) / 2;
            
            ctx.drawImage(videoElement, 
                0, 0, videoElement.videoWidth, videoElement.videoHeight,
                x, 0, newWidth, newHeight
            );

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                
                if (g > r * 1.5 && g > b * 1.5) {
                    data[i + 3] = 0;
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        
        isProcessing = false;
        
        if (isGreenScreenEnabled) {
            animationFrameId = requestAnimationFrame(processFrame);
        }
    }

    function startCheckingSpeakingStatus() {
        checkSpeakingInterval = setInterval(async () => {
            try {
                const response = await fetch('/is_speaking', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sessionid: parseInt(sessionId)
                    })
                });
                const result = await response.json();
                
                const avatarSelect = document.getElementById('avatarSelect');
                const voiceSelect = document.getElementById('voiceSelect');
                avatarSelect.disabled = result.data;
                voiceSelect.disabled = result.data;
                
                if (result.data) {
                    recordButton.disabled = true;
                    recordButton.style.opacity = "0.6";
                } else {
                    recordButton.disabled = false;
                    recordButton.style.opacity = "1";
                }
            } catch (error) {
                console.error('Error checking speaking status:', error);
            }
        }, 500);
    }

    function showPermissionModal() {
        return new Promise((resolve, reject) => {
            permissionModal.style.display = 'block';
            
            allowButton.onclick = async () => {
                permissionModal.style.display = 'none';
                resolve(true);
            };
            
            denyButton.onclick = () => {
                permissionModal.style.display = 'none';
                resolve(false);
            };
        });
    }

    async function checkAndRequestPermission() {
        try {
            const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
            
            if (permissionStatus.state === 'granted') {
                return true;
            } else if (permissionStatus.state === 'prompt') {
                const userAgreed = await showPermissionModal();
                if (!userAgreed) {
                    showError('對話功能需要麥克風權限才能使用。');
                    return false;
                }
                return true;
            } else if (permissionStatus.state === 'denied') {
                showError('麥克風權限已被封鎖。請在瀏覽器設定中更改權限設定。');
                return false;
            }
        } catch (err) {
            const userAgreed = await showPermissionModal();
            return userAgreed;
        }
    }

    function getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/mp4',
            'audio/mpeg',
            ''
        ];
        
        for (const type of types) {
            if (type === '' || MediaRecorder.isTypeSupported(type)) {
                console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' MediaRecorder supported type: ' + type);
                return type;
            }
        }
        console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' No supported MediaRecorder type found.');
        return '';
    }

    async function requestMicrophoneAccess() {
        try {
            const hasPermission = await checkAndRequestPermission();
            if (!hasPermission) {
                return;
            }

            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    sampleSize: 16
                },
                video: false
            });

            const mimeType = getSupportedMimeType();
            const options = {
                audioBitsPerSecond: 16000
            };
            if (mimeType) {
                options.mimeType = mimeType;
            }
            mediaRecorder = new MediaRecorder(audioStream, options);
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                await transcribeAudio(audioBlob);
                audioChunks = [];
            };

            startRecording();
            
        } catch (err) {
            console.error('Microphone access error:', err);
            let errorMsg = '無法存取麥克風。';
            
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                errorMsg = '請允許麥克風存取權限。';
            } else if (err.name === 'NotFoundError') {
                errorMsg = '找不到麥克風裝置。';
            } else if (err.name === 'NotReadableError') {
                errorMsg = '麥克風可能正被其他應用程式使用。';
            }
            
            showError(errorMsg);
            recordButton.textContent = '開始對話';
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    }
    
    function startRecording() {
        audioChunks = [];
        mediaRecorder.start(200);
        recordButton.textContent = '對話中...';
        recordButton.disabled = true;
        stopButton.disabled = false;
        transcriptionResult.textContent = '';
        console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' Recording started');
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recordButton.textContent = '開始對話';
            recordButton.disabled = false;
            stopButton.disabled = true;
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' Recording stopped');
            
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
        }
    }

    function updateChatDisplay() {
        chatHistory.innerHTML = '';
        const startIndex = Math.max(0, dialogHistory.length - 4);
        for (let i = startIndex; i < dialogHistory.length; i++) {
            const dialog = dialogHistory[i];
            const dialogElement = document.createElement('div');
            dialogElement.innerHTML = `
                <div class="chat-message user-message">${dialog.user}</div>
                <div class="chat-message ai-message">${dialog.ai}</div>
            `;
            chatHistory.appendChild(dialogElement);
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function humantalk(message) {
        $('#rtc_media_player').prop('muted', false);
        console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}`+ ' Sending to /human api : ' + message);
        
        try {
            await fetch('/human', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: message,
                    type: 'echo',
                    sessionid: parseInt(sessionId)
                })
            });
        } catch (error) {
            console.error('Error sending message:', error);
            showError('發送訊息失敗');
        }
    }

    async function transcribeAudio(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio.wav');
            formData.append('sessionid', sessionId);

            const response = await fetch('https://talk-dev.aitago.tw:9880/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('轉錄請求失敗');
            }
            
            const result = await response.json();
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' whisper STT result:', result.input_text);
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' ai answer result:', result.text);
            
            // 使用時間戳產生唯一ID
            const uniqueId = 'ai-response-' + Date.now();
            
            // 每次對話建立一個新的對話容器，包含使用者訊息和AI回應區域
            const dialogContainer = document.createElement('div');
            dialogContainer.className = 'dialog-pair';
            
            // 將用戶的輸入顯示在對話歷史中
            const userMessage = document.createElement('div');
            userMessage.innerHTML = `<div class="chat-message user-message">${result.input_text}</div>`;
            dialogContainer.appendChild(userMessage);
            
            // 建立打字指示器作為AI回應的預留位置
            const aiMessage = document.createElement('div');
            aiMessage.id = uniqueId;
            aiMessage.innerHTML = `
                <div class="chat-message ai-message typing-indicator">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            `;
            dialogContainer.appendChild(aiMessage);
            
            // 將整個對話容器添加到聊天歷史
            chatHistory.appendChild(dialogContainer);
            chatHistory.scrollTop = chatHistory.scrollHeight;
            
            // 清空transcriptionResult
            transcriptionResult.innerHTML = '';
            
            // 在HTML中添加當前對話的ID，用於API回調確認
            const speakingStatusElement = document.createElement('div');
            speakingStatusElement.id = 'speaking-status-indicator';
            speakingStatusElement.setAttribute('data-response-id', uniqueId);
            speakingStatusElement.style.display = 'none';
            document.body.appendChild(speakingStatusElement);
            
            // 發送請求給虛擬人
            await humantalk(result.text);
            
            // 監聽虛擬人實際開始說話的狀態
            const checkIfSpeaking = setInterval(async () => {
                try {
                    const response = await fetch('/is_speaking', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sessionid: parseInt(sessionId) })
                    });
                    const speakingStatus = await response.json();
                    
                    // 當虛擬人開始說話時，更新對話記錄
                    if (speakingStatus.data === true) {
                        clearInterval(checkIfSpeaking);
                        
                        // 找到並替換臨時的打字指示器為實際AI回應
                        const tempAiMessage = document.getElementById(uniqueId);
                        if (tempAiMessage) {
                            tempAiMessage.innerHTML = `<div class="chat-message ai-message">${result.text}</div>`;
                            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' 將...切換成AI回應:', result.text);
                        }
                        
                        // 保存完整對話歷史
                        const dialogSet = {
                            user: result.input_text,
                            ai: result.text
                        };
                        dialogHistory.push(dialogSet);
                        
                        // 這裡調用updateChatDisplay來限制顯示的對話數量
                        updateChatDisplay();
                        
                        // 清除臨時元素
                        document.body.removeChild(speakingStatusElement);
                    }
                } catch (error) {
                    console.error('Error checking speaking status:', error);
                }
            }, 200);

            // 設置超時處理，避免永遠等待
            setTimeout(() => {
                if (document.getElementById('speaking-status-indicator')) {
                    clearInterval(checkIfSpeaking);
                    
                    // 找到並替換臨時的打字指示器為實際AI回應
                    const tempAiMessage = document.getElementById(uniqueId);
                    if (tempAiMessage) {
                        tempAiMessage.innerHTML = `<div class="chat-message ai-message">${result.text}</div>`;
                    }
                    
                    // 保存完整對話歷史
                    const dialogSet = {
                        user: result.input_text,
                        ai: result.text
                    };
                    dialogHistory.push(dialogSet);
                    
                    // 清除臨時元素
                    document.body.removeChild(speakingStatusElement);
                }
            }, 5000);

        } catch (error) {
            console.error('處理錯誤:', error);
            showError('處理失敗: ' + error.message);
            transcriptionResult.textContent = '處理失敗';
        }
    }

    function startPlay() {
        $('#rtc_media_player').show();
        
        if (sdk) {
            sdk.close();
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        }
        sdk = new SrsRtcWhipWhepAsync();
        
        videoElement.srcObject = sdk.stream;

        videoElement.addEventListener('loadedmetadata', () => {
            if (!canvas) {
                initializeGreenScreen(videoElement);
            }
        });

        videoElement.addEventListener('playing', () => {
            if (isGreenScreenEnabled && !animationFrameId) {
                processFrame();
            }
        });

        const host = window.location.hostname;
        const streamName = sessionId === '0' ? 'livestream' : `livestream${sessionId}`;
        const url = `https://${host}:1986/rtc/v1/whep/?app=live&stream=${streamName}`;
        
        sdk.play(url).then(function(session) {
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` +' WebRTC stream connected successfully');
        }).catch(function(reason) {
            sdk.close();
            $('#rtc_media_player').hide();
            console.error('WebRTC stream error:', reason);
            showError('串流連接失敗: ' + reason);
        });
    }

    recordButton.onclick = () => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            requestMicrophoneAccess();
        }
    };
    
    stopButton.onclick = stopRecording;

    document.addEventListener('keydown', function(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        
        if (event.key.toLowerCase() === 'r') {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                requestMicrophoneAccess();
            }
        }
        
        if (event.key.toLowerCase() === 't') {
            stopRecording();
        }

        if (event.code === 'Space') {
            event.preventDefault();
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                requestMicrophoneAccess();
            } else {
                stopRecording();
            }
        }
    });

    chatHistory.addEventListener('touchstart', function(e) {
        e.stopPropagation();
    }, false);

    document.getElementById('voiceSelect').onchange = function() {
        const configType = parseInt(this.value);
        fetch('/switch_tts_endpoint', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sessionid: parseInt(sessionId),
                config_type: configType
            })
        });
        console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' Voice type changed to ' + configType);
    };

    document.getElementById('avatarSelect').onchange = function() {
        const avatar_id = this.value;
        fetch('/switch_avatar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sessionid: parseInt(sessionId),
                avatar_id: avatar_id
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.code === 0) {
                currentDisplayName = data.display_name;
            }
        })
        .then(() => {
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' Avatar changed to ' + avatar_id);
        })
        .catch(error => {
            console.error('Error switching avatar:', error);
            showError('切換虛擬人失敗');
        });
    };

    async function initializeUIWithConfig() {
        try {
            const configResponse = await fetch('/get_current_config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sessionid: parseInt(sessionId)
                })
            });
            const configData = await configResponse.json();
            
            if (configData.code === 0) {
                const avatarsResponse = await fetch('/get_avatars');
                const avatarsData = await avatarsResponse.json();
                
                const voicesResponse = await fetch('/get_voices');
                const voicesData = await voicesResponse.json();
                
                if (avatarsData.code === 0 && voicesData.code === 0) {
                    const avatarSelect = document.getElementById('avatarSelect');
                    const voiceSelect = document.getElementById('voiceSelect');
                    
                    avatarSelect.innerHTML = '';
                    Object.values(avatarsData.data).forEach(avatar => {
                        const option = document.createElement('option');
                        option.value = avatar.id;
                        option.textContent = avatar.display_name;
                        avatarSelect.appendChild(option);
                    });
                    
                    voiceSelect.innerHTML = '';
                    Object.values(voicesData.data).forEach(voice => {
                        const option = document.createElement('option');
                        option.value = voice.id;
                        option.textContent = voice.name;
                        voiceSelect.appendChild(option);
                    });
                    
                    avatarSelect.value = configData.data.current_avatar_id;
                    voiceSelect.value = configData.data.current_voice_type.toString();
                }
            }
            console.log(`${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}` + ' UI initialized');
        } catch (error) {
            console.error('Error initializing UI:', error);
            showError('初始化設定失敗');
        }
    }

    window.onbeforeunload = function() {
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
        }
        if (sdk) {
            sdk.close();
        }
        if (checkSpeakingInterval) {
            clearInterval(checkSpeakingInterval);
        }
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    };
    
    $(document).ready(async function() {
        $('#rtc_media_player').prop('muted', true);
        startCheckingSpeakingStatus();
        await initializeUIWithConfig();
        startPlay();
    });
});
