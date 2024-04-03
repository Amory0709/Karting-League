using System;
using KartGame.KartSystems;
using UnityEngine;

/// <summary>
/// This class inherits from TargetObject and represents a PickupObject.
/// </summary>
public class PickupObject : TargetObject
{
    [Header("PickupObject")]

    [Tooltip("New Gameobject (a VFX for example) to spawn when you trigger this PickupObject")]
    public GameObject spawnPrefabOnPickup;

    [Tooltip("Destroy the spawned spawnPrefabOnPickup gameobject after this delay time. Time is in seconds.")]
    public float destroySpawnPrefabDelay = 10;
    
    [Tooltip("Destroy this gameobject after collectDuration seconds")]
    public float collectDuration = 0f;

    void Start() {
        Register();
    }

    void OnCollect(ArcadeKart kartPlayer)
    {
        if (CollectSound)
        {
            AudioUtility.CreateSFX(CollectSound, transform.position, AudioUtility.AudioGroups.Pickup, 0f);
        }

        if (spawnPrefabOnPickup)
        {
            var vfx = Instantiate(spawnPrefabOnPickup, CollectVFXSpawnPoint.position, Quaternion.identity);
            Destroy(vfx, destroySpawnPrefabDelay);
        }
               
        Objective.OnUnregisterPickup(this, kartPlayer);

        TimeManager.OnAdjustTime(TimeGained);

        //Destroy(gameObject, collectDuration);
    }
    
    void OnTriggerEnter(Collider other)
    {
        if ((layerMask.value & 1 << other.gameObject.layer) > 0 && other.gameObject.CompareTag("Player"))
        {
            // Get the reference to the kart player script/component
            ArcadeKart kartPlayer = other.gameObject.GetComponentInParent<ArcadeKart>();

            if (kartPlayer != null)
            {
                var m_TimeManager = FindObjectOfType<TimeManager>();
                Debug.Log($"OnEnter checkpoint: {gameObject.name} {gameObject.gameObject.name}");
                var newRecord = new ObjectiveRecord
                {
                    ObjectiveType = ObjectiveType.CheckPoint,
                    Name = gameObject.name,
                    Time = TimeSpan.FromSeconds(m_TimeManager.TotalTime - m_TimeManager.TimeRemaining)
                };
                kartPlayer.UpdateRecords(newRecord);
                OnCollect(kartPlayer);
            }
        }
    }
}
