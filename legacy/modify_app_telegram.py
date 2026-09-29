import re
with open('d:/ews/frontend/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'<td class="px-4 py-3 text-right">\s*<button onclick="selectPatient\(\$\{p\.patient_id\}\)"'
replacement = r"""<td class="px-4 py-3 text-right flex items-center justify-end gap-2">
        ${p.state === 'ALERT' ? `<button onclick="sendTelegramAlert(${p.patient_id})" class="px-2.5 py-1 text-xs font-semibold bg-red-100 hover:bg-red-600 hover:text-white text-red-700 rounded transition" title="Send Telegram Alert"><i data-lucide="send" class="w-3 h-3 inline"></i> Telegram</button>` : ''}
        <button onclick="selectPatient(${p.patient_id})\""""

new_content = re.sub(pattern, replacement, content)

# Check if sendTelegramAlert exists, if not, append it
if 'function sendTelegramAlert' not in new_content:
    new_content += """
// --- Telegram Alert Logic ---
async function sendTelegramAlert(patientId) {
    try {
        const response = await fetch(`/api/telegram_alert?patient_id=${patientId}`, { method: 'POST' });
        const data = await response.json();
        if(data.success) {
            alert(`Telegram Alert Sent for Patient ${patientId}!`);
        } else {
            alert(`Failed to send Telegram alert: ${data.message}`);
        }
    } catch (err) {
        alert("Failed to reach server to send Telegram alert.");
    }
}
"""

with open('d:/ews/frontend/app.js', 'w', encoding='utf-8') as f:
    f.write(new_content)
