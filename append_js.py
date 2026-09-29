with open('d:/ews/frontend/app.js', 'a', encoding='utf-8') as f:
    f.write('''
// --- AI Chat Logic ---
let chatCsvFile = null;

document.getElementById('chat-csv-upload')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        chatCsvFile = file;
        document.getElementById('csv-upload-status').textContent = file.name;
    } else {
        chatCsvFile = null;
        document.getElementById('csv-upload-status').textContent = 'No file selected';
    }
});

function appendChatMessage(sender, text) {
    const container = document.getElementById('chat-history');
    if(!container) return;
    const isBot = sender === 'bot';
    
    const div = document.createElement('div');
    div.className = 'flex gap-3 ' + (isBot ? '' : 'flex-row-reverse');
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ' + 
        (isBot ? 'bg-cyan-100 text-cyan-700' : 'bg-slate-200 text-slate-700');
    iconDiv.innerHTML = isBot ? '<i data-lucide="bot" class="w-4 h-4"></i>' : '<i data-lucide="user" class="w-4 h-4"></i>';
    
    const msgDiv = document.createElement('div');
    msgDiv.className = 'bg-white border border-slate-200 rounded-2xl px-4 py-2 text-sm max-w-[85%] shadow-sm ' + 
        (isBot ? 'rounded-tl-none text-slate-800 whitespace-pre-wrap' : 'rounded-tr-none text-slate-800 whitespace-pre-wrap');
    msgDiv.textContent = text;
    
    div.appendChild(iconDiv);
    div.appendChild(msgDiv);
    container.appendChild(div);
    
    // Re-render icons for dynamically added content
    if(window.lucide) lucide.createIcons();
    
    container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg && !chatCsvFile) return;
    
    if (msg) appendChatMessage('user', msg);
    input.value = '';
    
    const formData = new FormData();
    if (msg) formData.append('message', msg);
    if (chatCsvFile) formData.append('file', chatCsvFile);
    
    try {
        // Clear file after sending so we don't keep uploading it
        if(chatCsvFile) {
            chatCsvFile = null;
            document.getElementById('chat-csv-upload').value = '';
            document.getElementById('csv-upload-status').textContent = 'No file selected';
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if(response.ok) {
            appendChatMessage('bot', data.reply);
        } else {
            appendChatMessage('bot', 'Error: ' + data.detail);
        }
    } catch (err) {
        appendChatMessage('bot', 'Failed to communicate with AI server.');
    }
}
''')
