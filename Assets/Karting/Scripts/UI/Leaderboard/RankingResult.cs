using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Networking;

namespace KartGame.UI.Leaderboard
{
    [System.Serializable]
    public class RankingResult
    {
        /// <summary>
        /// The round of the result
        /// </summary>
        public string round;

        /// <summary>
        /// Team scores
        /// </summary>
        public List<TeamScore> teamScores;

        /// <summary>
        ///     Is a valid raw channel request
        /// </summary>
        /// <returns></returns>
    }
    [System.Serializable]
    public class TeamScore
    {
        /// <summary>
        /// The Id to identify the team
        /// </summary>
        public string teamName;
        /// Score of the game
        /// </summary>
        public int score;

        public double timeCost;
    }
}
