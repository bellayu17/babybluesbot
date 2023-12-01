class Chatbox {
    constructor() {
        this.args = {
            msgBox: document.querySelector('.chatbox_msgbox'),
            sendButton: document.querySelector('.btn')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {msgBox, sendButton} = this.args;

        sendButton.addEventListener('click', () => this.onSendButton(msgBox))

        const node = msgBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(msgBox)
            }
        })
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", msg: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ msg: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "BabyBlues", msg: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "BabyBlues")
            {
                html += '<div class="txt_item txt_item--out">' + item.msg + '</div>'
            }
            else
            {
                html += '<div class="txt_item txt_item--in">' + item.msg + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox_txt');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();