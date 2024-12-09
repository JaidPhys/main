import streamlit as st

# Funzione principale
def main():
    # Stile personalizzato
    st.markdown(
        """
        <style>
        body {
            background-color: #fffaf0; /* Sfondo color avorio */
            color: #5a5a5a; /* Testo grigio scuro */
        }
        .metric-box {
            background-color: #fff5e6; /* Beige molto chiaro */
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            margin-bottom: 20px; /* Spazio tra i box */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Ombra leggera per il box */
        }
        .stButton>button {
            background-color: #f4a460; /* Sabbia chiara */
            color: white;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Titolo e descrizione
    st.title("I Costi Nascosti di Dispositivi Elettronici e Problemi Muscoloscheletrici")
    st.write(
        """
        Inserisci il numero di dipendenti della tua azienda per calcolare:
        - **Giorni spesi per rendere i dispositivi elettronici aziendali conformi con gli standard medici regolativi.**
        - **Giorni spesi dai dipendenti per completare formazione in fatto di salute in ufficio.**
        - **Giorni persi a causa di assenze per problemi muscolari e scheletrici.**
        - **Giorni persi per riduzione di produttività correlata ai problemi muscoloscheletrici.**
        """
    )

    # Input: Numero di dipendenti
    num_employees = st.number_input(
        "🔢 Inserisci il numero di dipendenti",
        min_value=1,
        step=1,
        value=10,
    )

    # Calcoli
    days_admin_dse = (num_employees * 10) / 60 / 8
    days_training_dse = (num_employees * 45) / 60 / 8
    msk_absence_rate = 0.026
    msk_absence_days = num_employees * msk_absence_rate * 20
    msk_pain_incidence = 0.15
    msk_productivity_loss = 0.15
    msk_loss_days = num_employees * msk_pain_incidence * msk_productivity_loss * 220

    # Risultati
    st.subheader("📊 Risultati")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <strong>Giorni per conformare i dispositivi:</strong>
                <br><span style="font-size: 1.5em; color: #d2691e;">{days_admin_dse:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-box">
                <strong>Giorni persi per produttività ridotta:</strong>
                <br><span style="font-size: 1.5em; color: #d2691e;">{msk_loss_days:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <strong>Giorni per formazione dei dipendenti:</strong>
                <br><span style="font-size: 1.5em; color: #d2691e;">{days_training_dse:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-box">
                <strong>Giorni persi per assenze:</strong>
                <br><span style="font-size: 1.5em; color: #d2691e;">{msk_absence_days:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("💡 I valori sono stimati considerando un giorno lavorativo di 8 ore e sono calcolati su base annua.")

# Avvio dell'app
if __name__ == "__main__":
    main()
