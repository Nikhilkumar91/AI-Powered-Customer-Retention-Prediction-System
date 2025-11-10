function closeModal(id) {
    document.getElementById(id).style.display = "none";
}

document.getElementById("aboutBtn").onclick = () => {
    document.getElementById("aboutModal").style.display = "flex";
};
document.getElementById("devBtn").onclick = () => {
    document.getElementById("devModal").style.display = "flex";
};
