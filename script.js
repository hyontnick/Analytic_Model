const allSideMenu = document.querySelectorAll('#sidebar .side-menu.top li a');

allSideMenu.forEach(item => {
	const li = item.parentElement;

	item.addEventListener('click', function () {
		allSideMenu.forEach(i => {
			i.parentElement.classList.remove('active');
		})
		li.classList.add('active');
	})
});




// TOGGLE SIDEBAR
const menuBar = document.querySelector('#content nav .bx.bx-menu');
const sidebar = document.getElementById('sidebar');

menuBar.addEventListener('click', function () {
	sidebar.classList.toggle('hide');
})



const searchButton = document.querySelector('#content nav form .form-input button');
const searchButtonIcon = document.querySelector('#content nav form .form-input button .bx');
const searchForm = document.querySelector('#content nav form');

searchButton.addEventListener('click', function (e) {
	if (window.innerWidth < 576) {
		e.preventDefault();
		searchForm.classList.toggle('show');
		if (searchForm.classList.contains('show')) {
			searchButtonIcon.classList.replace('bx-search', 'bx-x');
		} else {
			searchButtonIcon.classList.replace('bx-x', 'bx-search');
		}
	}
})


if (window.innerWidth < 768) {
	sidebar.classList.add('hide');
} else if (window.innerWidth > 576) {
	searchButtonIcon.classList.replace('bx-x', 'bx-search');
	searchForm.classList.remove('show');
}


window.addEventListener('resize', function () {
	if (this.innerWidth > 576) {
		searchButtonIcon.classList.replace('bx-x', 'bx-search');
		searchForm.classList.remove('show');
	}
})



const switchMode = document.getElementById('switch-mode');

switchMode.addEventListener('change', function () {
	if (this.checked) {
		document.body.classList.add('dark');
	} else {
		document.body.classList.remove('dark');
	}
})

var imageElement = document.getElementById("image");

function changeImage(imageSrc) {
	imageElement.src = imageSrc;
}

function showModel() {
	document.getElementById("models").style.display = "block";
	document.getElementById("visualisation").style.display = "none";
	document.getElementById("questions").style.display = "none";
	document.getElementById("storeInfo").style.display = "none";
}

function showVisualisation() {
	document.getElementById("models").style.display = "none";
	document.getElementById("visualisation").style.display = "block";
	document.getElementById("questions").style.display = "none";
	document.getElementById("storeInfo").style.display = "none";
}

function showQuestions() {
	document.getElementById("models").style.display = "none";
	document.getElementById("visualisation").style.display = "none";
	document.getElementById("questions").style.display = "block";
	document.getElementById("storeInfo").style.display = "none";
}

function showAnalyse() {
	document.getElementById("models").style.display = "none";
	document.getElementById("visualisation").style.display = "none";
	document.getElementById("questions").style.display = "none";
	document.getElementById("storeInfo").style.display = "block";
}

const searchInput = document.getElementById('search-input');
const todoLists = document.querySelectorAll('.todo-list');

searchInput.addEventListener('input', function (event) {
	const searchText = event.target.value.toLowerCase();

	todoLists.forEach(function (todoList) {
		const todoItems = todoList.getElementsByTagName('li');

		Array.from(todoItems).forEach(function (todoItem) {
			const text = todoItem.textContent.toLowerCase();

			if (text.includes(searchText)) {
				todoItem.style.display = 'block';
			} else {
				todoItem.style.display = 'none';
			}
		});
	});
});


function showSection(sectionId) {
	var sections = document.getElementById("questions").children;

	// Masquer toutes les sections de questions
	for (var i = 0; i < sections.length; i++) {
		sections[i].style.display = "none";
	}

	// Afficher la section cible
	var targetSection = document.getElementById(sectionId);
	targetSection.style.display = "block";

	// Faire défiler vers la section cible
	targetSection.parentElement.style.display = "block"; // Ajouter cette ligne pour afficher le conteneur parent de la section cible
	targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); // Modifier cette ligne pour définir le bloc de défilement comme "start"
}

// Définir votre clé d'API OpenAI
const apiKey = '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';

// Fonction pour envoyer une requête à l'API OpenAI et obtenir la réponse
async function obtenirReponse(question, conversation) {
	// Afficher la barre de progression
	document.getElementById('progress-bar-inner').style.width = '0%';
	document.getElementById('progress-bar').style.display = 'block';

	const response = await fetch('https://api.openai.com/v1/chat/completions', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `Bearer ${apiKey}`
		},
		body: JSON.stringify({
			model: 'gpt-3.5-turbo',
			messages: conversation
		})
	});

	const data = await response.json();
	const reply = data.choices[0].message.content;

	// Masquer la barre de progression
	document.getElementById('progress-bar').style.display = 'none';

	return reply;
}

// Fonction pour ajouter un message au chatbox
function ajouterMessage(role, nom, contenu) {
	const chatbox = document.getElementById('chatbox');
	const message = document.createElement('div');
	message.classList.add('message', role);
	const name = document.createElement('div');
	name.classList.add('name');
	name.textContent = nom;
	const content = document.createElement('div');
	content.classList.add('content');
	content.textContent = contenu;
	message.appendChild(name);
	message.appendChild(content);
	chatbox.appendChild(message);
	chatbox.scrollTop = chatbox.scrollHeight;
}

// Fonction pour afficher la réponse de manière progressive
async function afficherReponseProgressive(reponse) {
	const chatbox = document.getElementById('chatbox');
	const typingIndicator = document.createElement('span');
	typingIndicator.classList.add('typing-indicator');
	typingIndicator.textContent = '...';
	chatbox.appendChild(typingIndicator);
	chatbox.scrollTop = chatbox.scrollHeight;

	await delay(1000);

	chatbox.removeChild(typingIndicator);

	for (let i = 0; i < reponse.length; i++) {
		chatbox.lastChild.lastChild.textContent += reponse[i];
		chatbox.scrollTop = chatbox.scrollHeight;
		await delay(30);
	}
}

// Fonction pour ajouter un délai
function delay(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

// Gérer l'événement du bouton d'envoi
document.getElementById('send-button').addEventListener('click', async function () {
	const userInput = document.getElementById('user-input');
	const userQuestion = userInput.value.trim();

	if (userQuestion !== '') {
		// Désactiver le champ de saisie et le bouton d'envoi
		userInput.disabled = true;
		document.getElementById('send-button').disabled = true;

		// Ajouter l'indicateur de saisie
		document.getElementById('typing-indicator').classList.add('typing-indicator');

		// Ajouter la question de l'utilisateur au chatbox
		ajouterMessage('user-message', 'Utilisateur', userQuestion);

		// Créer la conversation avec la question de l'utilisateur
		const conversation = [
			{ role: 'system', content: 'You are a helpful assistant.' },
			{ role: 'user', content: userQuestion }
		];

		// Obtenir la réponse de l'assistant
		const assistantReply = await obtenirReponse(userQuestion, conversation);

		// Activer à nouveau le champ de saisie et le bouton d'envoi
		userInput.disabled = false;
		document.getElementById('send-button').disabled = false;

		// Supprimer l'indicateur de saisie
		document.getElementById('typing-indicator').classList.remove('typing-indicator');

		// Ajouter la réponse de l'assistant au chatbox
		ajouterMessage('assistant-message', 'AM Assistant', '');

		// Ajouter l'indicateur de réponse
		const responseIndicator = document.createElement('span');
		responseIndicator.classList.add('response-indicator');
		responseIndicator.textContent = '...'; //Assistant est en train de répondre
		document.getElementById('user-input-container').appendChild(responseIndicator);

		// Afficher la réponse de manière progressive
		await afficherReponseProgressive(assistantReply);

		// Supprimer l'indicateur de réponse
		document.getElementById('user-input-container').removeChild(responseIndicator);

		// Réinitialiser le champ de saisie de l'utilisateur
		userInput.value = '';

		// Activer le bouton de copie et stocker la réponse générée
		const copyButton = document.getElementById('copy-button');
		copyButton.disabled = false;
		copyButton.dataset.response = assistantReply;
	}
});

// Gérer l'événement du bouton de copie
document.getElementById('copy-button').addEventListener('click', function () {
	const copyButton = document.getElementById('copy-button');
	const response = copyButton.dataset.response;
	if (response) {
		navigator.clipboard.writeText(response)
			.then(() => {
				copyButton.textContent = 'Copié !';
				setTimeout(function () {
					copyButton.textContent = 'Copier la réponse';
				}, 3000);
			})
			.catch(err => {
				console.error('Erreur lors de la copie :', err);
			});
	}
});
