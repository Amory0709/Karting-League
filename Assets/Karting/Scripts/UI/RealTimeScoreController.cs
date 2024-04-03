using System.Collections.Generic;
using System.Linq;
using KartGame.KartSystems;
using TMPro;
using UnityEngine;

public class RealTimeScoreController : MonoBehaviour
{
    public GameObject playerScoreTemplate;
    public GameObject scrollViewContent;

    private List<GameObject> playerScores = new List<GameObject>();
    private ArcadeKart[] Karts = new ArcadeKart[] { };

    private const string Name = "Name";
    private const string Score = "Score";
    private const string Ranking = "Ranking";

    // Start is called before the first frame update
    void Start()
    {
        AddScoreHeader();

        Karts = FindObjectsOfType<ArcadeKart>();
        foreach (var kart in Karts.OrderBy(k => k.PlayerName))
        {
            GameObject item = Instantiate(playerScoreTemplate);
            item.transform.SetParent(scrollViewContent.transform);
            item.transform.Find(Ranking).GetComponent<TMP_Text>().text = "1";
            item.transform.Find(Name).GetComponent<TMP_Text>().text = kart.PlayerName;
            item.transform.Find(Score).GetComponent<TMP_Text>().text = "0";
            playerScores.Add(item);
        }
    }

    private void AddScoreHeader()
    {
        var headerNameAndDisplayNames = new Dictionary<string, string>()
        {
            { Ranking, "Ranking"},
            { Name, "Player"},
            { Score, "Score"}
        };

        GameObject header = Instantiate(playerScoreTemplate);
        header.transform.SetParent(scrollViewContent.transform);
        foreach (var pair in headerNameAndDisplayNames)
        {
            var tmpText = header.transform.Find(pair.Key).GetComponent<TMP_Text>();
            tmpText.text = pair.Value;
            ChangeStyle(tmpText);
        }
    }

    private void ChangeStyle(TMP_Text tMP_Text)
    {
        //tMP_Text.color = new Color32(107, 53, 84, 255);
        tMP_Text.fontSize = 12;
        tMP_Text.fontWeight = FontWeight.Regular;
    }

    // Update is called once per frame
    void Update()
    {
        var karts = FindObjectsOfType<ArcadeKart>();
        if (karts.All(k => k.Score == 0))
        {
            return;
        }

        var orderedKarts = karts.OrderByDescending(k => k.Score).ThenBy(k => k.TimeCost).ToList();
        for (var i = 0; i < orderedKarts.Count; i++)
        {
            var kart = orderedKarts[i];

            playerScores[i].transform.Find(Ranking).GetComponent<TMP_Text>().text = (i + 1).ToString();
            playerScores[i].transform.Find(Name).GetComponent<TMP_Text>().text = kart.PlayerName;
            playerScores[i].transform.Find(Score).GetComponent<TMP_Text>().text = kart.Score.ToString();
        }
    }
}
